//! Flare owns shadow-session lifecycle, routing transport and presentation.
//! Compiled prediction plugins execute on bounded threads inside the serving process.

mod tui;

use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    fs,
    io::IsTerminal,
    net::TcpListener,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex, atomic::Ordering},
    thread,
    time::{Duration, Instant},
};

pub type Failure = Box<dyn std::error::Error + Send + Sync>;
type Shared = Arc<Mutex<View>>;

#[derive(Default)]
struct View {
    phase: String,
    model: String,
    gpu: String,
    output: PathBuf,
    responses: Vec<String>,
    report: Option<Value>,
    error: Option<String>,
    done: bool,
}

#[derive(Deserialize)]
struct Experiment {
    binary: PathBuf,
    package: PathBuf,
    package_id: String,
    catalog: PathBuf,
    hardware: PathBuf,
    gpu: String,
    port: u16,
}

pub fn agent(seconds: u64) -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(seconds)))
        .build()
        .into()
}
pub fn get(client: &ureq::Agent, url: &str) -> Result<Value, Failure> {
    Ok(serde_json::from_str(
        &client.get(url).call()?.body_mut().read_to_string()?,
    )?)
}
pub fn save(path: &Path, value: &Value) -> Result<(), Failure> {
    let temp = path.with_extension("tmp");
    fs::write(&temp, serde_json::to_vec_pretty(value)?)?;
    fs::rename(temp, path)?;
    Ok(())
}
fn read(path: &Path) -> Result<Value, Failure> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}
fn phase(view: &Shared, text: &str) {
    view.lock().unwrap().phase = text.into();
}
fn cancelled() -> Result<(), Failure> {
    if crate::STOP.load(Ordering::Relaxed) {
        Err("Stopped by operator; capture remains incomplete".into())
    } else {
        Ok(())
    }
}

fn gpu_guard(uuid: &str) -> Result<(), Failure> {
    if !uuid.starts_with("GPU-") {
        return Err("experiment requires an explicit GPU UUID".into());
    }
    for _ in 0..2 {
        let processes = Command::new("nvidia-smi")
            .args(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"])
            .output()?;
        if !processes.status.success() {
            return Err("cannot inspect GPU ownership".into());
        }
        if String::from_utf8_lossy(&processes.stdout)
            .lines()
            .any(|l| l.split(',').next().is_some_and(|x| x.trim() == uuid))
        {
            return Err("assigned GPU has a compute process; Parallax will not stop it".into());
        }
        let info = Command::new("nvidia-smi")
            .args([
                "--id",
                uuid,
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()?;
        let text = String::from_utf8(info.stdout)?;
        let values: Vec<u64> = text
            .trim()
            .split(',')
            .map(|x| x.trim().parse())
            .collect::<Result<_, _>>()?;
        if !info.status.success() || values.len() != 2 || values[0] >= 512 || values[1] != 0 {
            return Err("assigned GPU is not idle".into());
        }
        thread::sleep(Duration::from_secs(1));
    }
    Ok(())
}

fn wait_child(child: &mut Child, timeout: Duration) -> Result<std::process::ExitStatus, Failure> {
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if start.elapsed() > timeout {
            return Err(format!("process {} exceeded shutdown budget", child.id()).into());
        }
        thread::sleep(Duration::from_millis(50));
    }
}
const GAUGES: [&str; 5] = [
    "flare_active_requests",
    "flare_pending_requests",
    "flare_retiring_request_slots",
    "flare_http_inflight_requests",
    "flare_frontend_active_workers",
];
fn is_drained(text: &str) -> bool {
    GAUGES.iter().all(|name| {
        text.lines().any(|line| {
            let mut fields = line.split_whitespace();
            fields.next() == Some(*name)
                && fields.next().and_then(|x| x.parse::<f64>().ok()) == Some(0.)
        })
    })
}
fn drain(url: &str) -> Result<(), Failure> {
    let client = agent(2);
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(30) {
        let text = client
            .get(&format!("{url}/metrics"))
            .call()?
            .body_mut()
            .read_to_string()?;
        if is_drained(&text) {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(100));
    }
    Err("owned server did not drain; left running for inspection".into())
}

fn session(config: &Experiment, output: &Path, requests: u8, view: &Shared) -> Result<(), Failure> {
    phase(view, "Verifying package and experiment");
    let executable = std::env::current_exe()?.canonicalize()?;
    if executable != config.binary.canonicalize()? {
        return Err("experiment binary differs from this Flare executable".into());
    }
    let inspect = Command::new(&executable)
        .arg("inspect")
        .arg(&config.package)
        .arg("--json")
        .output()?;
    if !inspect.status.success() {
        return Err(String::from_utf8_lossy(&inspect.stderr).into_owned().into());
    }
    let info: Value = serde_json::from_slice(&inspect.stdout)?;
    if info["package"]["signatures_verified"] != true
        || info["package"]["package_id"] != config.package_id
        || info["package"]["tensor_parallel"] != 1
    {
        return Err("package signature, identity or single-rank contract mismatch".into());
    }
    let catalog = read(&config.catalog)?;
    if !catalog["identity"]
        .as_str()
        .is_some_and(|s| s.ends_with(&format!(";package={}", config.package_id)))
    {
        return Err("expert catalog is not bound to this package".into());
    }
    let model = info["model"]["model_id"]
        .as_str()
        .ok_or("package model identity missing")?;
    view.lock().unwrap().model = model.into();
    if config.port < 1024 {
        return Err("Parallax requires an unprivileged loopback port".into());
    }
    drop(TcpListener::bind(("127.0.0.1", config.port))?);
    gpu_guard(&config.gpu)?;
    cancelled()?;
    let profile = output.join("profile.json");
    save(
        &profile,
        &json!({"version":1,"catalog":config.catalog,"hardware":config.hardware,"output":output.join("capture"),
        "predictors":[{"name":"recent","config":{}},{"name":"ewma","config":{"alpha":0.5}},{"name":"frequency","config":{"warmup_frames":128}}],
        "evaluation":{"version":2,"top_k_per_layer":32,"max_promotions_per_forecast":128},
        "target_offset":8,"horizon_ns":60_000_000,"budget_ns":1_000_000_000u64,"max_frames":2048}),
    )?;
    let key = output.join("empty-api-key");
    fs::write(&key, "")?;
    let log = fs::File::create(output.join("server.log"))?;
    let mut server = Command::new(&executable)
        .arg("run")
        .arg(&config.package)
        .args([
            "--gpus",
            "0",
            "--host",
            "127.0.0.1",
            "--port",
            &config.port.to_string(),
            "--served-model-name",
            model,
        ])
        .arg("--api-key-file")
        .arg(&key)
        .args([
            "--verify-weights",
            "--embedding-table-storage",
            "ram",
            "--scheduler-policy",
            "balanced",
            "--max-concurrency",
            "32",
            "--mtp-draft-tokens",
            "0",
            "--no-mtp-adaptive-verification",
            "--prefix-cache-memory",
            "0",
            "--prefix-cache-disk-limit",
            "0",
            "--prefix-cache-drain-on-shutdown=off",
            "--parallax-routing",
        ])
        .arg("--parallax-profile")
        .arg(&profile)
        .env("CUDA_VISIBLE_DEVICES", &config.gpu)
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .process_group(0)
        .spawn()?;
    save(
        &output.join("launch.json"),
        &json!({"pid":server.id(),"package_id":config.package_id,"complete":false}),
    )?;
    let url = format!("http://127.0.0.1:{}", config.port);
    let mut ready = false;
    let result = (|| -> Result<(), Failure> {
        phase(view, "Loading verified weights and prepared graphs");
        let start = Instant::now();
        loop {
            cancelled()?;
            if let Some(status) = server.try_wait()? {
                return Err(format!("server exited {status}; see server.log").into());
            }
            if get(&agent(2), &format!("{url}/health")).is_ok() {
                break;
            }
            if start.elapsed() > Duration::from_secs(600) {
                return Err("server readiness timed out".into());
            }
            thread::sleep(Duration::from_millis(250));
        }
        let listener = Command::new("ss")
            .args(["-ltnp", "sport", "=", &format!(":{}", config.port)])
            .output()?;
        if !String::from_utf8_lossy(&listener.stdout).contains(&format!("pid={},", server.id())) {
            return Err("endpoint is not owned by this Parallax session".into());
        }
        ready = true;
        let metadata = get(
            &agent(2),
            &format!("{url}/v1/parallax/routing?after={}", u64::MAX),
        )?;
        if metadata["model_id"] != model || metadata["enabled"] != true {
            return Err("native model identity or tracing activation mismatch".into());
        }
        observe(config, output, &url, model, requests, view)?;
        Ok(())
    })();
    phase(view, "Draining owned server");
    let cleanup = (|| -> Result<(), Failure> {
        if server.try_wait()?.is_none() {
            if ready {
                drain(&url)?;
            }
            // SAFETY: this PID belongs to the child created above. Flare drains on SIGTERM.
            unsafe {
                libc::kill(server.id() as i32, libc::SIGTERM);
            }
        }
        let status = wait_child(&mut server, Duration::from_secs(120))?;
        save(
            &output.join("launch.json"),
            &json!({"pid":server.id(),"package_id":config.package_id,"server_returncode":status.code(),"complete":result.is_ok() && status.success()}),
        )?;
        if !status.success() {
            return Err(format!("owned server exited {status}").into());
        }
        Ok(())
    })();
    match (result, cleanup) {
        (Err(original), Err(cleanup)) => {
            Err(format!("{original}; cleanup: {cleanup}; owned PID {}", server.id()).into())
        }
        (Err(error), _) | (_, Err(error)) => Err(error),
        _ => {
            let capture = output.join("capture/capture.json");
            let report =
                flare_helios::server::parallax::replay(&capture).map_err(|e| e.to_string())?;
            if report != read(&output.join("capture/comparison.json"))? {
                return Err("native offline replay differs from live summary".into());
            }
            if report["routing_frames"].as_u64().unwrap_or(0) == 0
                || ["recent", "ewma", "frequency"]
                    .iter()
                    .any(|name| report["workers"][name]["completed"].as_u64().unwrap_or(0) == 0)
            {
                return Err("capture has no routing or missing provider predictions".into());
            }
            view.lock().unwrap().report = Some(report);
            Ok(())
        }
    }
}

fn observe(
    _config: &Experiment,
    output: &Path,
    url: &str,
    model: &str,
    requests: u8,
    view: &Shared,
) -> Result<(), Failure> {
    let prompts = [
        "Explain GPU memory and expert routing to a software engineer.",
        "Plan a weekend walk through a city.",
        "Explain how to debug a concurrency problem.",
    ];
    let mut responses = Vec::new();
    for index in 0..requests {
        cancelled()?;
        phase(
            view,
            &format!(
                "Serving request {} / {requests} · in-process predictors",
                index + 1
            ),
        );
        let body = json!({"model":model,"prompt":format!("Demonstration {}. {}",index+1,prompts[index as usize%prompts.len()]),
            "max_tokens":128,"ignore_eos":true,"temperature":1.0,"top_p":0.95,"top_k":-1,"seed":1234});
        let encoded = serde_json::to_string(&body)?;
        let response: Value = serde_json::from_str(
            &agent(30)
                .post(&format!("{url}/v1/completions"))
                .header("Content-Type", "application/json")
                .send(encoded.as_str())?
                .body_mut()
                .read_to_string()?,
        )?;
        if response["usage"]["completion_tokens"] != 128
            || response["choices"].as_array().map(Vec::len) != Some(1)
        {
            return Err("incomplete response token counts".into());
        }
        view.lock().unwrap().responses.push(
            response["choices"][0]["text"]
                .as_str()
                .unwrap_or("")
                .to_owned(),
        );
        responses.push(response);
        save(&output.join("responses.json"), &json!(responses))?;
    }
    drain(url)?;
    Ok(())
}

pub fn run(args: crate::cli::ParallaxArgs) -> Result<(), Failure> {
    if let Some(path) = args.replay {
        let alternate = args.evaluation_config.is_some();
        let report = if let Some(config) = args.evaluation_config {
            flare_helios::server::parallax::reevaluate(&path, serde_json::from_slice(&fs::read(config)?)?)?
        } else {
            let report = flare_helios::server::parallax::replay(&path)?;
            if report != read(&path.with_file_name("comparison.json"))? {
                return Err("native replay differs from saved comparison".into());
            }
            report
        };
        if !args.plain && std::io::stderr().is_terminal() {
            let view = Arc::new(Mutex::new(View {
                phase: if alternate { "Recorded forecasts · alternate evaluation" } else { "Saved capture · exact replay verified" }.into(),
                model: "Recorded native Parallax session".into(),
                gpu: "Replay · no GPU".into(),
                output: path.parent().unwrap_or(Path::new(".")).to_owned(),
                report: Some(report),
                done: true,
                ..View::default()
            }));
            tui::run(&view)?;
        } else {
            println!("{}", serde_json::to_string_pretty(&report)?);
        }

        eprintln!("{}", if alternate { "Recorded forecasts evaluated with the requested configuration; source unchanged." } else { "Native replay exactly reproduces the saved comparison." });
        return Ok(());
    }
    let config: Experiment =
        serde_json::from_slice(&fs::read(args.experiment.ok_or("experiment required")?)?)?;
    let output = if let Some(path) = args.output {
        path
    } else {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        std::env::temp_dir().join(format!("flare-parallax-{stamp}-{}", std::process::id()))
    };
    fs::create_dir(&output)?;
    let view = Arc::new(Mutex::new(View {
        phase: "Preparing session".into(),
        gpu: config.gpu.clone(),
        output: output.clone(),
        ..View::default()
    }));
    // SAFETY: the handler only stores to the same atomic stop flag as normal serving.
    unsafe {
        libc::signal(libc::SIGINT, crate::stop as *const () as usize);
        libc::signal(libc::SIGTERM, crate::stop as *const () as usize);
    }
    let worker_view = view.clone();
    let worker = thread::spawn(move || {
        let result = session(&config, &output, args.requests, &worker_view);
        let mut state = worker_view.lock().unwrap();
        state.error = result.as_ref().err().map(ToString::to_string);
        state.phase = if result.is_ok() {
            "Complete · replay verified · server drained"
        } else {
            "Incomplete · evidence retained"
        }
        .into();
        state.done = true;
        let _ = save(
            &output.join("demo.json"),
            &json!({"complete":result.is_ok(),"error":state.error,"package_id":config.package_id,
            "responses":state.responses.len(),"routing_frames":state.report.as_ref().map(|x|&x["routing_frames"]),"replay_verified":result.is_ok()}),
        );
        result.map_err(|e| e.to_string())
    });
    let display = if !args.plain && std::io::stderr().is_terminal() {
        tui::run(&view)
    } else {
        tui::plain(&view)
    };
    if display.is_err() {
        crate::STOP.store(true, Ordering::Relaxed);
    }
    let result = worker
        .join()
        .map_err(|_| "Parallax session thread panicked")?;
    display?;
    eprintln!(
        "Parallax evidence: {}",
        view.lock().unwrap().output.display()
    );
    result.map_err(Into::into)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn drain_requires_every_admission_and_retirement_gauge() {
        let text = GAUGES.map(|name| format!("{name} 0\n")).concat();
        assert!(is_drained(&text));
        assert!(!is_drained(&text.replace(
            "flare_retiring_request_slots 0",
            "flare_retiring_request_slots 1"
        )));
        assert!(!is_drained(
            &text.replace("flare_frontend_active_workers 0\n", "")
        ));
    }
}
