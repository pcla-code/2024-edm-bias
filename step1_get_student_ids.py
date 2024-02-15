"""
Purpose: Determine student IDs that are both in Canvas Database and in Demographics Google Sheet.

Configuration on which the code was tested: Python 3.10 on Windows 11 (with 8 GB of RAM).
"""


def read_ids(filename: str) -> set[int]:
    """
    Get all unique user IDs exported either from the Canvas DB or Demographics GSheet.
    :param filename Name of the input TXT file containing 1 ID per row.
    :return A set of int IDs from the TXT file.
    """
    unique_ids = set()
    with open(filename) as text_file:
        for row in text_file:
            if 'canvas' in filename:
                student_id = int(row[-6:-1])  # For Canvas DB data, get the last 5 digits
            else:
                student_id = int(row)  # For Demographics GSheet data, get the ID as-is
            unique_ids.add(student_id)
    return unique_ids


def write_intersection_ids(filename='ids-intersection.txt') -> None:
    """
    Output a sorted list of IDs that overlap in both data sources.
    :param filename Name of the output TXT file containing 1 ID per row.
    """
    canvas_ids = read_ids('ids-canvas.txt')  # metadata_user_id column in the Canvas DB
    gsheet_ids = read_ids('ids-gsheet.txt')  # Canvas ID column in the Demographics GSheet
    gsheet_ids_remove = read_ids('ids-to-remove.txt')  # No match for the region
    gsheet_ids = gsheet_ids.difference(gsheet_ids_remove)
    intersection_ids = sorted(list(canvas_ids.intersection(gsheet_ids)))
    with open(filename, 'w') as text_file:
        for student_id in intersection_ids:
            text_file.write(str(student_id) + '\n')


if __name__ == '__main__':
    write_intersection_ids()
