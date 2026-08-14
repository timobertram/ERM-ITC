"""Parser for ITC2007 Track 3 (Curriculum-Based Course Timetabling) .ctt files.

Format (see comp01.ctt for a concrete example):

    Name: <name>
    Courses: <n>
    Rooms: <n>
    Days: <n>
    Periods_per_day: <n>
    Curricula: <n>
    Constraints: <n>

    COURSES:
    <course_id> <teacher_id> <n_lectures> <min_working_days> <n_students>
    ...

    ROOMS:
    <room_id> <capacity>
    ...

    CURRICULA:
    <curriculum_id> <n_courses> <course_id_1> ... <course_id_n>
    ...

    UNAVAILABILITY_CONSTRAINTS:
    <course_id> <day> <period>
    ...

    END.

Spec reference: docs/itc2007.pdf in https://github.com/Docheinstein/itc2007-cct
(mirrors the official ITC2007 Track 3 problem description).
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Course:
    id: str
    teacher_id: str
    n_lectures: int
    min_working_days: int
    n_students: int


@dataclass
class Room:
    id: str
    capacity: int


@dataclass
class Curriculum:
    id: str
    course_ids: List[str]


@dataclass
class UnavailabilityConstraint:
    course_id: str
    day: int
    period: int


@dataclass
class CTTProblem:
    name: str
    n_days: int
    periods_per_day: int
    courses: Dict[str, Course] = field(default_factory=dict)
    rooms: Dict[str, Room] = field(default_factory=dict)
    curricula: Dict[str, Curriculum] = field(default_factory=dict)
    unavailability: List[UnavailabilityConstraint] = field(default_factory=list)

    @property
    def n_periods(self) -> int:
        return self.n_days * self.periods_per_day


def parse_ctt(path: str) -> CTTProblem:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]

    header = {}
    i = 0
    while ":" in lines[i] and not lines[i].endswith(":"):
        key, _, value = lines[i].partition(":")
        header[key.strip()] = value.strip()
        i += 1

    problem = CTTProblem(
        name=header["Name"],
        n_days=int(header["Days"]),
        periods_per_day=int(header["Periods_per_day"]),
    )
    n_constraints = int(header["Constraints"])

    section = None
    for line in lines[i:]:
        if line in ("COURSES:", "ROOMS:", "CURRICULA:", "UNAVAILABILITY_CONSTRAINTS:"):
            section = line[:-1]
            continue
        if line == "END.":
            break

        parts = line.split()
        if section == "COURSES":
            cid, teacher, n_lect, min_days, n_students = parts
            problem.courses[cid] = Course(cid, teacher, int(n_lect), int(min_days), int(n_students))
        elif section == "ROOMS":
            rid, capacity = parts
            problem.rooms[rid] = Room(rid, int(capacity))
        elif section == "CURRICULA":
            qid, n_courses = parts[0], int(parts[1])
            course_ids = parts[2:2 + n_courses]
            problem.curricula[qid] = Curriculum(qid, course_ids)
        elif section == "UNAVAILABILITY_CONSTRAINTS":
            cid, day, period = parts
            problem.unavailability.append(UnavailabilityConstraint(cid, int(day), int(period)))

    assert len(problem.unavailability) == n_constraints, (
        f"expected {n_constraints} unavailability constraints, parsed {len(problem.unavailability)}"
    )
    return problem


def summarize(problem: CTTProblem) -> str:
    n_lectures_total = sum(c.n_lectures for c in problem.courses.values())
    curricula_sizes = [len(q.course_ids) for q in problem.curricula.values()]
    return "\n".join([
        f"problem: {problem.name}  ({problem.n_days} days x {problem.periods_per_day} periods/day "
        f"= {problem.n_periods} periods)",
        f"courses: {len(problem.courses)}  (total lectures: {n_lectures_total})",
        f"rooms: {len(problem.rooms)}",
        f"curricula: {len(problem.curricula)}  (size min={min(curricula_sizes)} max={max(curricula_sizes)})",
        f"unavailability constraints: {len(problem.unavailability)}",
    ])


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ITC2007/real/comp01.ctt"
    print(summarize(parse_ctt(path)))
