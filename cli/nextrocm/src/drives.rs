use nextrocm_models::{registry as models, Config};
use nextrocm_serve::drives;
pub fn not_moving(cfg: &Config, id: &str) -> Result<(), String> {
    match drives::moving_why(&models::registry(cfg), id) {
        Some(why) => Err(why),
        None => Ok(()),
    }
}
