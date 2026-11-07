use super::*;
/// A failed replacement may already have unmapped its old backing. Restore the
/// failed binding as well as every earlier binding before declaring recovery.
pub(super) fn publish(
    bindings: &[Binding],
    mut replace: impl FnMut(usize, Location) -> Result<()>,
) -> Result<()> {
    for (index, binding) in bindings.iter().enumerate() {
        if let Err(original) = replace(binding.region, binding.to) {
            for old in bindings[..=index].iter().rev() {
                if let Err(rollback) = replace(old.region, old.from) {
                    return Err(Error::Poisoned(format!(
                        "mapping update failed ({original}); rollback failed ({rollback})"
                    )));
                }
            }
            return Err(Error::Backend(format!(
                "mapping update rolled back: {original}"
            )));
        }
    }
    Ok(())
}
