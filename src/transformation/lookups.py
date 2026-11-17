"""
Official DfT STATS19 Road Casualty Statistics Lookup Mappings.
Provides human-readable decodes for codes used in accident, weather, road, light, and surface attributes.
Source: Department for Transport (DfT) Road Casualty Statistics Data Guide.
"""

SEVERITY_MAP = {
    1: "Fatal",
    2: "Serious",
    3: "Slight",
    -1: "Unknown"
}

WEATHER_MAP = {
    1: "Fine no high winds",
    2: "Raining no high winds",
    3: "Snowing no high winds",
    4: "Fine + high winds",
    5: "Raining + high winds",
    6: "Snowing + high winds",
    7: "Fog or mist",
    8: "Other",
    9: "Unknown",
    -1: "Missing"
}

ROAD_TYPE_MAP = {
    1: "Roundabout",
    2: "One way street",
    3: "Dual carriageway",
    6: "Single carriageway",
    7: "Slip road",
    9: "Unknown",
    12: "One way street/Slip road",
    -1: "Missing"
}

LIGHT_CONDITIONS_MAP = {
    1: "Daylight",
    4: "Darkness - lights lit",
    5: "Darkness - lights unlit",
    6: "Darkness - no lighting",
    7: "Darkness - lighting unknown",
    -1: "Missing"
}

ROAD_SURFACE_MAP = {
    1: "Dry",
    2: "Wet or damp",
    3: "Snow",
    4: "Frost or ice",
    5: "Flood over 3cm deep",
    6: "Oil or diesel",
    7: "Mud",
    9: "Unknown",
    -1: "Missing"
}

URBAN_RURAL_MAP = {
    1: "Urban",
    2: "Rural",
    3: "Unallocated",
    -1: "Missing"
}

def decode_attribute(value, lookup_dict, default="Unknown"):
    """Safely map integer lookup codes to descriptive text labels."""
    try:
        val_int = int(value)
        return lookup_dict.get(val_int, default)
    except (ValueError, TypeError):
        return default
