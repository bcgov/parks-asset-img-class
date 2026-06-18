from .image_loader import load_asset_images_base64
from .model_router import run_model

def predict_asset_attributes(asset_id,
                             df,
                             model_name,
                             prompt,
                             image_root="data/processed/images_clean",
                             provider="auto",
                             max_tokens=4096):
    """
    Top-level function for use in production.
    Loads images, builds prompt, calls model, returns raw JSON string.
    """
    images = load_asset_images_base64(asset_id, df, image_root)

    if len(images) == 0:
        return {
            "asset_id": asset_id,
            "error": "No images found",
            "response": None
        }

    try:
        result = run_model(
            model_name,
            prompt,
            images,
            provider=provider,
            max_tokens=max_tokens,
        )
        return {
            "asset_id": asset_id,
            "error": None,
            "response": result
        }

    except Exception as e:
        return {
            "asset_id": asset_id,
            "error": str(e),
            "response": None
        }
