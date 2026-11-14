"""Rule-based recommendation engine (turns scores into coaching text).
The scores themselves come from the ML model; these rules only explain them.
An LLM could later rewrite this text more naturally, but is NOT needed for the assessment."""
from .config import pretty

STRONG, WEAK = 80, 65

STRENGTH = {
    "balance": "Good body balance; your weight stays well over your base.",
    "footwork": "Footwork is efficient and well timed.",
    "posture": "Posture is upright and controlled throughout the movement.",
    "head_stability": "Your head stays steady, which helps you track the ball.",
    "follow_through": "Strong and complete follow-through.",
    "body_rotation": "Good use of shoulder and hip rotation to generate power.",
    "wrist_action": "Good wrist action at contact.",
    "action_execution": "The overall shot/stroke is executed cleanly.",
}

ADVICE = {
    "balance": ("Body balance is unstable during the movement.",
                "Keep your head over your base and avoid leaning too far; practise slow shadow movements."),
    "footwork": ("Foot placement or movement is inconsistent.",
                 "Move the front foot decisively to the line of the ball/shuttle and keep a stable base on landing."),
    "posture": ("Posture changes too much during the action.",
                "Keep the back straight with a slight forward lean and bend the knees a little more."),
    "head_stability": ("The head moves too much during the shot.",
                       "Keep your eyes level and your head still until contact; practise with a fixed focus point."),
    "follow_through": ("The follow-through is short or incomplete.",
                       "Let the hands finish high and complete the swing instead of stopping at contact."),
    "body_rotation": ("Limited shoulder/hip rotation.",
                      "Rotate hips first and let the shoulders follow to add power."),
    "wrist_action": ("Wrist movement at contact is weak or late.",
                     "Use a relaxed grip and snap the wrist through contact."),
    "action_execution": ("The overall technique needs refinement.",
                         "Practise the shot in slow motion, then gradually increase speed while keeping the form."),
}


def band(score):
    return "Excellent" if score >= 85 else "Good" if score >= STRONG else "Average" if score >= WEAK else "Needs work"


def generate_feedback(scores, overall):
    strengths = [STRENGTH[p] for p, s in sorted(scores.items(), key=lambda x: -x[1])
                 if s >= STRONG and p in STRENGTH][:3]
    weak = sorted([(p, s) for p, s in scores.items() if s < WEAK and p in ADVICE], key=lambda x: x[1])
    improvements = [ADVICE[p][0] for p, _ in weak]
    recommendations = [ADVICE[p][1] for p, _ in weak]
    if not weak:  # nothing is clearly weak -> suggest improving the lowest parameter
        p = min(scores, key=scores.get)
        if p in ADVICE:
            improvements.append(f"Lowest area: {pretty(p)} ({scores[p]:.0f}/100).")
            recommendations.append(ADVICE[p][1])
    summary = (f"Overall technique: {band(overall)} ({overall:.0f}/100). "
               + (f"Main strength: {pretty(max(scores, key=scores.get))}. " if scores else "")
               + (f"Main area to improve: {pretty(min(scores, key=scores.get))}." if scores else ""))
    return {"summary": summary, "strengths": strengths,
            "improvements": improvements, "recommendations": recommendations}
