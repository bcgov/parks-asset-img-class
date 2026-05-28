# src/vlm/prompts.py
"""
Prompt templates for VLM zero-shot attribute prediction.
"""

#update as needed
# Asset-specific prompts

STAIRS_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single stair asset, identify the most likely
    attribute values. For each of the following attributes, the possible values are
    given below. Predict exactly ONE value from the listed options for each
    attribute, and provide a confidence score (0.0-1.0) for each prediction.

    Attributes to predict:
    - fall_height: low (<0.5m) | medium (0.5-1.2m) | high (>1.2m)
    - has_pedestrian_railing: 2 railings | 1 railing | No railings
    - material_frame_tank_body: PVC | Gravel | Natural Surface | Earth-Filled |
                                Aluminum | Metal | Steel | Rock/Stone | Concrete |
                                Box Step | Timber/Wood
    - number_of_steps: few (<10) | medium (10-20) | many (>20)
    - structure_position: Elevated | At-Grade | Other

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "<attribute_key>": {
        "value": "<predicted value or 'unable to determine'>",
        "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine an attribute from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

TRAIL_BRIDGE_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single trail bridge asset, identify the most 
    likely attribute values. For each of the following attributes, the possible values 
    are given below. Predict exactly ONE value from the listed options for each
    attribute, and provide a confidence score (0.0-1.0) for each prediction.

    Attributes to predict:
    - abutment_material: Aluminum Sill Fill | Composite | Concrete |
                         Gabions | Steel | Timber
    - bridge_type: Beam | Fallen Tree | Suspension | Truss | Other
    - decking_material: Aluminum | Asphalt | Composite | Concrete |
                        Steel | Timber
    - fall_height: low (<1.2m) | medium (1.2-5m) | high (>5m)
    - has_pedestrian_railing: 2 railings | 1 railing | No railings
    - length: short (<6m) | medium (6-20m) | long (>20m)
    - width: narrow (<0.9m) | standard (0.9-1.5m) | wide (>1.5m)
    - structure_position: Elevated | At-Grade | Other

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "<attribute_key>": {
        "value": "<predicted value or 'unable to determine'>",
        "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine an attribute from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

# Attribute-specific prompts

STRUCTURE_POSITION_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.
    
    Using ALL provided images for a single asset, identify the most likely 
    structure position.
    
    Predict exactly ONE value from the listed options:
    - Elevated
    - At-Grade
    - Other
    
    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "structure_position": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }
    
    If you cannot determine the structure position from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

PEDESTRIAN_RAILING_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.
    
    Using ALL provided images of this single asset, identify whether it has 
    a pedestrian railing and how many.
    
    Predict exactly ONE value from the listed options:
    - 2 railings
    - 1 railing
    - No railings
    
    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "has_pedestrian_railing": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }
    
    If you cannot determine from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

#Dynamic prompts 
#needed since bins have different ranges per asset type 

def make_length_prompt(asset_type):
    return f"""
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single {asset_type} asset, estimate its length.

    Predict exactly ONE value from the listed options for this asset type:

    Boardwalk < 1.2m High: short (<20m) | medium (20-100m) | long (>100m)
    Boardwalk > 1.2m High: short (<10m) | medium (10-30m) | long (>30m)
    Stairs: short (<5m) | medium (5-20m) | long (>20m)
    Trail Bridge: short (<6m) | medium (6-20m) | long (>20m)
    Viewing Platform: small (<10m) | medium (10-20m) | large (>20m)

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {{
        "length_bin": {{"value": "<bin label>", "confidence": 0.85}}
    }}

    If you cannot determine the length, set value to "unable to determine" and confidence to 0.0.
    """


def make_width_prompt(asset_type):
    return f"""
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single {asset_type} asset, estimate its width.

    Predict exactly ONE value from the listed options for this asset type:

    Boardwalk < 1.2m High: narrow (<0.9m) | standard (0.9-1.5m) | wide (>1.5m)
    Boardwalk > 1.2m High: narrow (<0.9m) | standard (0.9-1.5m) | wide (>1.5m)
    Stairs: narrow (<0.8m) | standard (>=0.8m)
    Trail Bridge: narrow (<0.9m) | standard (0.9-1.5m) | wide (>1.5m)
    Viewing Platform: narrow (<3m) | medium (3-7m) | wide (>7m)

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {{
        "width_bin": {{"value": "<bin label>", "confidence": 0.85}}
    }}

    If you cannot determine the width, set value to "unable to determine" and confidence to 0.0.
    """

STEPS_BIN_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single stair asset, estimate the number of steps.

    Predict exactly ONE value from the listed options:
    - few (<10)
    - medium (10-20)
    - many (>20)

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "number_of_steps": {
            "value": "<bin label>",
            "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine the number of steps from the images, set value to
    "unable to determine" and confidence to 0.0.
    """
def make_fall_height_prompt(asset_type):
    
    if asset_type == "Viewing Platform":
        bins = "low (<1.2m) | medium (1.2-15m) | high (>15m)"
    elif asset_type == "Trail Bridge":
        bins = "low (<1.2m) | medium (1.2-5m) | high (>5m)"
    else:  # Boardwalks and Stairs
        bins = "low (<0.5m) | medium (0.5-1.2m) | high (>1.2m)"
    
    return f"""
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single {asset_type} asset, estimate the fall height.
    Fall height is the vertical distance from the asset surface to the ground below.

    Predict exactly ONE value from the listed options for this asset type:
    {bins}

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {{
        "fall_height": {{"value": "<bin label>", "confidence": <float 0.0-1.0>}}
    }}

    If you cannot determine the fall height from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

MATERIAL_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single stair asset, identify the material
    of the frame, tank, or body of the structure.

    Predict exactly ONE value from the listed options:
    - Timber/Wood
    - Concrete
    - Box Step
    - Rock/Stone
    - Metal
    - Steel
    - Aluminum
    - Earth-Filled
    - Natural Surface
    - Gravel
    - PVC

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "material_frame_tank_body": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine the material from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

DECKING_MATERIAL_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single asset, identify the decking material.

    Predict exactly ONE value from the listed options:
    - Timber
    - Steel
    - Aluminum
    - Composite
    - Concrete
    - Asphalt

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "decking_material": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine the decking material from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

STRUCTURE_MATERIAL_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single asset, identify the structure material.

    Predict exactly ONE value from the listed options:
    - Timber
    - Steel
    - Aluminum
    - Concrete
    - Stone

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "structure_material": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine the structure material from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

ABUTMENT_MATERIAL_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single Trail Bridge asset, identify the
    abutment material. The abutment is the structure at each end of the bridge
    that supports it and transfers loads to the ground.

    Predict exactly ONE value from the listed options:
    - Timber
    - Concrete
    - Steel
    - Gabions
    - Aluminum Sill Fill
    - Composite

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "abutment_material": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine the abutment material from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

BRIDGE_TYPE_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single Trail Bridge asset, identify the
    bridge type based on its structural design.

    Predict exactly ONE value from the listed options:
    - Beam
    - Truss
    - Suspension
    - Fallen Tree
    - Other

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "bridge_type": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine the bridge type from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

HAS_EDGE_GUARD_PROMPT_V1 = """
    You are an expert in park infrastructure analysis.

    Using ALL provided images of this single asset, identify whether it has
    an edge guard. An edge guard is a barrier along the edge of the structure
    that prevents people or objects from falling off the side.

    Predict exactly ONE value from the listed options:
    - Yes
    - No

    Return ONLY a valid JSON object with this exact schema (no markdown, no prose):
    {
        "has_edge_guard": {
            "value": "<predicted value or 'unable to determine'>",
            "confidence": <float 0.0-1.0>
        }
    }

    If you cannot determine whether there is an edge guard from the images, set value to
    "unable to determine" and confidence to 0.0.
    """

# Prompt registry
#update after generating prompts for attribute/asset

PROMPT_REGISTRY = {
    "stairs_v1": STAIRS_PROMPT_V1,
    "trail_bridge_v1": TRAIL_BRIDGE_PROMPT_V1,
    "structure_position_v1": STRUCTURE_POSITION_PROMPT_V1,
    "pedestrian_railing_v1": PEDESTRIAN_RAILING_PROMPT_V1,
    "steps_bin_v1": STEPS_BIN_PROMPT_V1,
    "material_v1": MATERIAL_PROMPT_V1,
    "decking_material_v1": DECKING_MATERIAL_PROMPT_V1,
    "structure_material_v1": STRUCTURE_MATERIAL_PROMPT_V1,
    "abutment_material_v1": ABUTMENT_MATERIAL_PROMPT_V1,
    "bridge_type_v1": BRIDGE_TYPE_PROMPT_V1,
    "has_edge_guard_v1": HAS_EDGE_GUARD_PROMPT_V1,
    "length_v1": make_length_prompt,
    "width_v1": make_width_prompt,
    "fall_height_v1": make_fall_height_prompt,
}