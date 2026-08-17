"""Generate a synthetic IT support ticket dataset for LoRA fine-tuning.

Produces realistic ServiceNow-style tickets across 5 categories with
priority labels and a short free-text severity justification (the field
the LLM-as-judge grades later). Ambiguity is intentional: ~12% of tickets
are deliberately cross-category or edge-case so the classifier (and the
eval harness) has real failure modes to surface, not a trivially easy task.

Usage:
    python data/generate_dataset.py --n 900 --seed 42 --out data/tickets_all.jsonl
"""

import argparse
import json
import random

CATEGORIES = ["Network", "Access/Password", "Hardware", "Software", "Billing"]
PRIORITIES = ["P1-Critical", "P2-High", "P3-Medium", "P4-Low"]

# Realistic priority distribution: most tickets are routine, few are critical.
PRIORITY_WEIGHTS = {"P1-Critical": 0.08, "P2-High": 0.24, "P3-Medium": 0.42, "P4-Low": 0.26}

DEPARTMENTS = ["Finance", "Sales", "Engineering", "HR", "Legal", "Marketing",
               "Operations", "Customer Support", "Procurement", "IT"]

NAMES = ["Priya Nair", "James Ortega", "Wei Chen", "Sofia Marín", "Daniel Kim",
         "Amara Okafor", "Liam O'Brien", "Fatima Al-Sayed", "Noah Becker",
         "Elena Petrova", "Ravi Iyer", "Grace Nguyen", "Tomas Novak", "Aisha Bello"]

SYSTEMS = ["Salesforce", "Workday", "SAP Concur", "the internal ERP", "Jira",
           "Confluence", "the VPN client", "Outlook 365", "the shared drive",
           "the payroll portal", "Zoom", "the ticketing system itself"]

DEVICES = ["Dell Latitude 5440", "MacBook Pro 14\"", "HP EliteBook 840",
           "Lenovo ThinkPad X1", "iPhone 14 (company-issued)",
           "Dell 27\" monitor", "docking station", "wireless headset",
           "conference room display"]


def _weighted_priority(rng):
    return rng.choices(list(PRIORITY_WEIGHTS.keys()), weights=list(PRIORITY_WEIGHTS.values()))[0]


def _justification(category, priority, detail, rng):
    impact = {
        "P1-Critical": [
            "This is blocking the entire team from working and has no workaround.",
            "Production-impacting with no fallback available; escalate immediately.",
            "Multiple users affected simultaneously with a hard business deadline today.",
        ],
        "P2-High": [
            "Single user is fully blocked from a core work task, but a workaround may exist.",
            "Affects one critical function; needs same-day attention to avoid delay.",
            "Time-sensitive but limited to one person or one system.",
        ],
        "P3-Medium": [
            "Inconvenient but the user has a workaround and can continue working.",
            "Non-blocking issue affecting productivity but not urgent.",
            "Should be fixed this week; not stopping any deadline-critical work.",
        ],
        "P4-Low": [
            "Cosmetic or minor annoyance with no impact on ability to work.",
            "Low urgency request, can be scheduled whenever convenient.",
            "Nice-to-have fix, no deadline pressure.",
        ],
    }
    return f"{detail} {rng.choice(impact[priority])}"


def _network_ticket(rng):
    name, dept = rng.choice(NAMES), rng.choice(DEPARTMENTS)
    scenarios = [
        ("VPN connection keeps dropping every few minutes",
         "vpn-drop", "P2-High",
         f"{name} in {dept} cannot maintain a stable VPN session, interrupting remote work."),
        ("No internet connectivity in the east wing since this morning",
         "wifi-outage", "P1-Critical",
         f"Entire {dept} floor has no network access; nobody on that floor can work."),
        ("Wi-Fi signal is weak near the third floor conference rooms",
         "wifi-weak", "P4-Low",
         "Signal is usable elsewhere on the floor; only affects one meeting room."),
        ("Cannot reach the internal file share from home office",
         "vpn-share", "P2-High",
         f"{name} needs the share for a client deliverable due today."),
        ("DNS resolution failing intermittently for internal tools",
         "dns-fail", "P3-Medium",
         "Affects several internal URLs; refreshing sometimes resolves it."),
        ("New office network jack in cubicle 214 is not active",
         "jack-inactive", "P4-Low",
         "User has a working alternate jack two desks away."),
        ("Site-to-site connection between HQ and the Austin office is down",
         "site-link-down", "P1-Critical",
         "Both offices lose access to shared internal systems until restored."),
    ]
    subj, tag, prio, detail = rng.choice(scenarios)
    body = (f"Hi IT, I'm {name} from {dept}. {subj}. "
            f"This started roughly {rng.choice(['this morning','an hour ago','yesterday afternoon','just now'])}. "
            f"I've already tried {rng.choice(['restarting my machine','reconnecting the VPN client','toggling Wi-Fi off and on','power-cycling the router'])} "
            f"but the issue persists.")
    return subj, body, "Network", prio, _justification("Network", prio, detail, rng)


def _access_ticket(rng):
    name, dept = rng.choice(NAMES), rng.choice(DEPARTMENTS)
    sysname = rng.choice(SYSTEMS)
    scenarios = [
        ("Locked out of my account after too many failed login attempts",
         "P2-High", f"{name} cannot access {sysname} at all and has work due today."),
        ("Password reset email never arrived",
         "P3-Medium", "User can still access most tools; only this one reset is pending."),
        ("MFA app reinstalled and now push notifications aren't arriving",
         "P1-Critical", f"{name} cannot log into any company system without MFA, total lockout."),
        ("Need access permissions added to a shared folder for a new project",
         "P4-Low", "Not urgent; project kickoff is next week."),
        ("Former contractor's account still shows as active",
         "P2-High", "Security exposure — access should have been revoked already."),
        ("Requesting elevated admin rights on my laptop for a one-time install",
         "P3-Medium", "Installation is optional and can wait for the next maintenance window."),
        ("Password expired and the self-service reset portal times out",
         "P2-High", f"{name} is blocked from logging in and reset portal itself is broken."),
    ]
    subj, prio, detail = rng.choice(scenarios)
    body = (f"Hello, this is {name} ({dept}). {subj}, related to {sysname}. "
            f"I need this resolved so I can {rng.choice(['submit an expense report','join a client call','finish a deployment','process payroll','access shared documents'])}.")
    return subj, body, "Access/Password", prio, _justification("Access/Password", prio, detail, rng)


def _hardware_ticket(rng):
    name, dept = rng.choice(NAMES), rng.choice(DEPARTMENTS)
    device = rng.choice(DEVICES)
    scenarios = [
        (f"{device} won't power on at all", "P2-High",
         f"{name} has no working machine and cannot do any work until replaced or repaired."),
        (f"{device} battery drains from 100% to 0% in under an hour", "P3-Medium",
         "Laptop still works while plugged in, so user can continue on-site."),
        (f"{device} screen has a hairline crack in the corner", "P4-Low",
         "Fully functional, purely cosmetic."),
        (f"{device} is making a loud fan noise under normal load", "P3-Medium",
         "Not blocking work but concerning for hardware longevity."),
        (f"{device} was stolen from a locked car overnight", "P1-Critical",
         f"{name} needs an emergency loaner and a security/data-wipe response today."),
        (f"{device} keyboard has several unresponsive keys", "P3-Medium",
         "Slows typing but an external keyboard is available as a stopgap."),
        (f"{device} won't connect to the docking station's second monitor", "P4-Low",
         "Single-monitor setup still works fine in the meantime."),
    ]
    subj, prio, detail = rng.choice(scenarios)
    body = (f"Hi, {name} here from {dept}. {subj}. "
            f"Serial/asset tag on file if needed. "
            f"{rng.choice(['This is affecting a client deadline.','No immediate deadline but wanted to flag it.','Please advise on loaner options.','Happy to bring it by the help desk.'])}")
    return subj, body, "Hardware", prio, _justification("Hardware", prio, detail, rng)


def _software_ticket(rng):
    name, dept = rng.choice(NAMES), rng.choice(DEPARTMENTS)
    sysname = rng.choice(SYSTEMS)
    scenarios = [
        (f"{sysname} crashes every time I try to export a report", "P2-High",
         f"{name} cannot deliver a report due this afternoon because of the crash."),
        (f"{sysname} is showing a blank screen after the latest update", "P1-Critical",
         f"Entire {dept} team relies on {sysname} and it's completely unusable post-update."),
        (f"Requesting installation of a licensed design tool", "P4-Low",
         "New hire onboarding item, not time-critical."),
        (f"{sysname} search results are noticeably slower than last week", "P3-Medium",
         "Annoying but usable; workflow continues at reduced speed."),
        (f"Recurring calendar invites in {sysname} are duplicating", "P3-Medium",
         "Cosmetic clutter, doesn't block scheduling."),
        (f"{sysname} keeps signing me out every 10 minutes", "P2-High",
         f"{name} loses unsaved work repeatedly, disrupting a deadline-driven task."),
        (f"Need a minor UI preference changed in {sysname}", "P4-Low",
         "Pure convenience request."),
    ]
    subj, prio, detail = rng.choice(scenarios)
    body = (f"Hi team, {name} from {dept}. {subj}. "
            f"Version info and a screenshot of the error are attached. "
            f"{rng.choice(['Started right after the last patch.','Happens consistently, every time.','Only happens intermittently, maybe 1 in 5 tries.'])}")
    return subj, body, "Software", prio, _justification("Software", prio, detail, rng)


def _billing_ticket(rng):
    name, dept = rng.choice(NAMES), rng.choice(DEPARTMENTS)
    scenarios = [
        ("Software license renewal invoice shows a discrepancy vs. the quoted price",
         "P3-Medium", "Needs correction before the invoice is paid, but payment isn't due for two weeks."),
        ("Duplicate charge appeared on the corporate card for the same SaaS subscription",
         "P2-High", "Finance needs this resolved before month-end close."),
        ("Requesting a cost breakdown of IT spend by department for budgeting",
         "P4-Low", "Informational request, no deadline."),
        ("Vendor is threatening to suspend service over an unpaid invoice IT never approved",
         "P1-Critical", "Active vendor risk with service suspension imminent today."),
        ("New employee needs a software license assigned and billed to their cost center",
         "P3-Medium", "Standard onboarding task, expected within the week."),
        ("Expense report for a hardware purchase was rejected due to missing PO number",
         "P3-Medium", "Blocks reimbursement but not urgent operationally."),
        ("Annual cloud hosting invoice is significantly higher than the approved budget",
         "P2-High", "Finance needs an explanation before approving payment this week."),
    ]
    subj, prio, detail = rng.choice(scenarios)
    body = (f"Hello, {name} from {dept} here. {subj}. "
            f"Reference number and relevant screenshots are attached. "
            f"{rng.choice(['Please advise before the next payment cycle.','This needs finance sign-off.','Flagging for visibility, not urgent.'])}")
    return subj, body, "Billing", prio, _justification("Billing", prio, detail, rng)


# Deliberately ambiguous / cross-category tickets — realistic edge cases that
# keep the task from being trivially separable, and give the eval harness
# genuine failure modes to analyze later.
def _ambiguous_ticket(rng):
    name, dept = rng.choice(NAMES), rng.choice(DEPARTMENTS)
    scenarios = [
        ("VPN login rejects my password even though I just reset it",
         "Access/Password", "P2-High",
         f"{name} is blocked from remote work; could be read as Network or Access issue."),
        ("Laptop won't join the office Wi-Fi after a Windows update",
         "Hardware", "P3-Medium",
         "Could plausibly be filed as Network or Software; classifying as Hardware since the update is device-side."),
        ("Can't install the VPN client because I don't have admin rights on my machine",
         "Access/Password", "P3-Medium",
         "Touches Network, Hardware, and Access — the root blocker is a permissions issue."),
        ("Billing software won't launch after this morning's security patch",
         "Software", "P2-High",
         "Overlaps Billing and Software; classifying by the actual defect, which is software crash."),
        ("Shared printer shows 'access denied' for my department",
         "Access/Password", "P3-Medium",
         "Could be Hardware (printer) or Access (permissions) — root cause is permissions."),
        ("Expense system login works but every report I submit shows the wrong department budget",
         "Billing", "P3-Medium",
         "Could be Software or Billing — classifying by business impact, which is budget/billing accuracy."),
    ]
    subj, cat, prio, detail = rng.choice(scenarios)
    body = (f"Hi, {name} from {dept}. {subj}. Not sure who owns this, sending to the general IT queue.")
    return subj, body, cat, prio, _justification(cat, prio, detail, rng)


GENERATORS = {
    "Network": _network_ticket,
    "Access/Password": _access_ticket,
    "Hardware": _hardware_ticket,
    "Software": _software_ticket,
    "Billing": _billing_ticket,
}


def generate(n, seed):
    rng = random.Random(seed)
    n_ambiguous = round(n * 0.12)
    n_regular = n - n_ambiguous
    per_category = n_regular // len(CATEGORIES)
    remainder = n_regular - per_category * len(CATEGORIES)

    records = []
    ticket_id = 10000
    for i, category in enumerate(CATEGORIES):
        count = per_category + (1 if i < remainder else 0)
        for _ in range(count):
            subj, body, cat, prio, justification = GENERATORS[category](rng)
            records.append(_make_record(ticket_id, subj, body, cat, prio, justification))
            ticket_id += 1

    for _ in range(n_ambiguous):
        subj, body, cat, prio, justification = _ambiguous_ticket(rng)
        records.append(_make_record(ticket_id, subj, body, cat, prio, justification))
        ticket_id += 1

    rng.shuffle(records)
    return records


def _make_record(ticket_id, subject, body, category, priority, justification):
    return {
        "ticket_id": f"INC{ticket_id}",
        "subject": subject,
        "description": body,
        "label": {
            "category": category,
            "priority": priority,
            "severity_justification": justification,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/tickets_all.jsonl")
    args = parser.parse_args()

    records = generate(args.n, args.seed)
    with open(args.out, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} tickets to {args.out}")
    from collections import Counter
    cat_counts = Counter(r["label"]["category"] for r in records)
    prio_counts = Counter(r["label"]["priority"] for r in records)
    print("Category distribution:", dict(cat_counts))
    print("Priority distribution:", dict(prio_counts))


if __name__ == "__main__":
    main()
