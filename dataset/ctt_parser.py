from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Course:
    """Course with lecture count, teacher, min working days and students."""
    id: str
    teacher_id: str
    n_lectures: int
    min_working_days: int
    n_students: int


@dataclass
class Room:
    """Room and its capacity."""
    id: str
    capacity: int


@dataclass
class Curriculum:
    """Curriculum: courses that must not overlap."""
    id: str
    course_ids: List[str]


@dataclass
class UnavailabilityConstraint:
    """Course unavailable at (day, period)."""
    course_id: str
    day: int
    period: int


@dataclass
class CTTProblem:
    """One ITC2007 instance."""
    name: str
    n_days: int
    periods_per_day: int
    courses: Dict[str, Course] = field(default_factory=dict)
    rooms: Dict[str, Room] = field(default_factory=dict)
    curricula: Dict[str, Curriculum] = field(default_factory=dict)
    unavailability: List[UnavailabilityConstraint] = field(default_factory=list)

    @property
    def n_periods(self) -> int:
        """Days times periods per day."""
        return self.n_days * self.periods_per_day


def parse_ctt(path: str) -> CTTProblem:
    """Parse an ITC2007 .ctt file."""
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
            problem.curricula[qid] = Curriculum(qid, parts[2:2 + n_courses])
        elif section == "UNAVAILABILITY_CONSTRAINTS":
            cid, day, period = parts
            problem.unavailability.append(UnavailabilityConstraint(cid, int(day), int(period)))

    assert len(problem.unavailability) == int(header["Constraints"])
    return problem
