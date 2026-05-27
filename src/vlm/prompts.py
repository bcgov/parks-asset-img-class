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

# Prompt registry
#update after generating prompts for attribute/asset

PROMPT_REGISTRY = {
    "stairs_v1": STAIRS_PROMPT_V1,
    "trail_bridge_v1": TRAIL_BRIDGE_PROMPT_V1,
    "structure_position_v1": STRUCTURE_POSITION_PROMPT_V1,
    "pedestrian_railing_v1": PEDESTRIAN_RAILING_PROMPT_V1,
}