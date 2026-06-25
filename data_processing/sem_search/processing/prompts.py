"""
Captioning prompts for the Gemma-4 VLM.

SYSTEM_PROMPT sets the annotator style; CAPTION_PROMPT_MULTI captions a 3-frame window
(with an optional ego-motion sentence injected into its {ego_motion_line} slot 
"""

SYSTEM_PROMPT = (
    "You are a driving scene annotator for a dashcam retrieval system. "
    "Captions are embedded for semantic search, so every sentence must describe "
    "something positively present in the image — searchable content only. "
    "Mentioning absent things pollutes the search index, so skip any category "
    "that does not apply and move on without comment. "
    "Describe only what you can identify with confidence: "
    "only call something a traffic light if you clearly see a signal housing with "
    "an illuminated red, green, or yellow lens; otherwise do not mention it. "
    "Only state a vehicle's color if lighting makes it unambiguous; otherwise "
    "describe the vehicle without color. "
    "Read sign text only if legible; otherwise call it an unreadable sign.\n\n"
    "Example of correct style for a sparse scene:\n"
    "'A two-lane rural road with a double yellow centerline curves right through "
    "dense trees. A silver sedan travels in the oncoming lane at mid-distance. "
    "The road surface is dry under overcast daytime light.'\n"
    "Note the example simply omits lights, signs, and pedestrians rather than "
    "remarking on them."
)

CAPTION_PROMPT_MULTI = (
    "You are given 3 dashcam frames from the same driving sequence, spaced "
    "~2.5 seconds apart. "
    "{ego_motion_line}"
    "Write one paragraph of continuous prose describing the scene and what "
    "happens across the sequence, as a single continuous scene — never reference "
    "frame numbers. Begin directly with the scene itself. "
    "Cover, in order, whichever of the following are actually present: "
    "(1) Road type — choose the best fit: highway, on-ramp/off-ramp, urban street, "
    "urban intersection, residential street, rural two-lane road, dirt or gravel "
    "road, off-road trail, parking lot, parking garage, bridge, tunnel, "
    "T-junction, 4-way stop, roundabout — plus lane count and markings. "
    "(2) Each visible vehicle: type, color if clearly discernible, lane position, "
    "distance (near, mid-distance, far), and how it moves across the sequence "
    "(e.g. a white SUV ahead pulls away, an oncoming truck passes). "
    "(3) Any illuminated traffic lights (red, green, yellow), including changes. "
    "(4) Any signs, text read verbatim. "
    "(5) Road surface: dry, wet, light snow cover, heavy snow cover, ice, slush, "
    "debris, or construction zone with cones or barriers. "
    "(6) Weather and lighting: clear, overcast, rain, light snow falling, heavy "
    "snow, fog, daytime, dusk, dawn, nighttime, glare. "
    "(7) Any emergency vehicles or unusual hazards. "
    "(8) Any pedestrians or cyclists, with position and action. "
    "(9) End with one clause on ego motion, phrased from the telemetry above if "
    "provided, otherwise inferred from how the scene changes (stationary, moving "
    "forward slowly, moving forward, accelerating, decelerating, turning left, "
    "turning right). "
    "Keep the paragraph under 150 words."
)


def _format_multi_prompt(ego_motion_line: str = "") -> str:
    """Fill the {ego_motion_line} slot. ego_motion_line is a sentence built from
    telemetry (see telemetry.build_ego_motion_line); empty string => the model falls
    back to inferring motion from the frames (prompt item 9)."""
    slot = (ego_motion_line.strip() + " ") if ego_motion_line and ego_motion_line.strip() else ""
    return CAPTION_PROMPT_MULTI.format(ego_motion_line=slot)
