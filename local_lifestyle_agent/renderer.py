from __future__ import annotations

from .schemas import FinalPlan, NormalizedIntent, RenderedOutput


class Renderer:
    def to_markdown(self, intent: NormalizedIntent, plan: FinalPlan) -> RenderedOutput:
        md = []
        md.append(f"# Sunday Afternoon Tea Plan — {intent.city}")
        md.append("")
        md.append(f"**Time Window:** {intent.time_window.get('day','Sunday')} {intent.time_window.get('start_local')}–{intent.time_window.get('end_local')}")
        md.append(f"**Travel Limit:** {intent.max_travel_minutes} minutes")
        if intent.origin_latlng:
            md.append(f"**Origin:** {intent.origin_latlng}")
        md.append("")

        md.append("## Primary Pick")
        md.append(f"**{plan.primary.name}**  \n{plan.primary.address}")
        md.append("")
        md.append("Why it fits:")
        for r in plan.primary.rationale:
            md.append(f"- {r}")

        md.append("")
        md.append("## Backups")
        for i, b in enumerate(plan.backups, 1):
            md.append(f"{i}. **{b.name}** — {b.address}")
            for r in b.rationale:
                md.append(f"   - {r}")

        md.append("")
        md.append("## Suggested Schedule")
        md.append(f"- Arrive: {plan.schedule.get('arrive_at')}")
        md.append(f"- Leave: {plan.schedule.get('leave_at')}")

        md.append("")
        md.append("## Tips")
        for t in plan.tips:
            md.append(f"- {t}")

        md.append("")
        md.append("## Assumptions")
        for a in plan.assumptions:
            md.append(f"- {a}")

        return RenderedOutput(markdown="\n".join(md))
