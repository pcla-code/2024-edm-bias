"""
Purpose: Create a data set of features that can be directly used in regression/classification.

Configuration on which the code was tested: Python 3.10 on Windows 11 (with 8 GB of RAM).
"""

# Data pre-processing
import csv
from datetime import datetime
from collections import defaultdict

# Data post-processing
import pandas as pd
from statistics import mean, stdev
import matplotlib.pyplot as plt
import seaborn as sns

SEMESTER1_START = datetime(2021, 2, 10)
SEMESTER1_END = datetime(2021, 6, 5, 23, 59, 59)
SEMESTER2_START = datetime(2021, 8, 26)
SEMESTER2_END = datetime(2021, 12, 18, 23, 59, 59)
SEMESTER3_START = datetime(2022, 1, 31)
SEMESTER3_END = datetime(2022, 5, 28, 23, 59, 59)


# ===== DEFINITION OF CLASSES THAT REPRESENT THE CANVAS DATA SET =====

class Event:
    """An abstract class representing an event (one row) in the Canvas DB."""
    def __init__(self, student_id: int, timestamp: datetime):
        assert 2 <= len(str(student_id)) <= 5
        assert SEMESTER1_START <= timestamp <= SEMESTER3_END
        self.student_id = student_id
        self.timestamp = timestamp


class EventAssetAccessed(Event):
    def __init__(self, student_id, timestamp, asset_id: int, asset_type: str, asset_category: str):
        assert all(value is not None for value in [asset_id, asset_type, asset_category])
        super().__init__(student_id, timestamp)
        self.asset_id = asset_id
        self.asset_type = asset_type
        self.asset_category = asset_category


class EventAttachmentCreated(Event):
    def __init__(self, student_id, timestamp, attachment_id: int):
        super().__init__(student_id, timestamp)
        self.attachment_id = attachment_id


class EventAttachmentUpdated(Event):
    def __init__(self, student_id, timestamp, attachment_id: int):
        super().__init__(student_id, timestamp)
        self.attachment_id = attachment_id


class EventDiscussionEntryCreated(Event):
    def __init__(self, student_id, timestamp, entry_id: int, parent_entry_id: int, length: int):
        assert all(value is not None for value in [entry_id, parent_entry_id, length])
        assert length > 0
        super().__init__(student_id, timestamp)
        self.entry_id = entry_id
        self.parent_entry_id = parent_entry_id
        self.length = length


class EventLoggedIn(Event):
    def __init__(self, student_id, timestamp):
        super().__init__(student_id, timestamp)


class EventQuizSubmitted(Event):
    def __init__(self, student_id, timestamp):
        super().__init__(student_id, timestamp)


class EventSubmissionCreated(Event):
    def __init__(self, student_id, timestamp, late: str):
        assert late in ['true', 'false']
        super().__init__(student_id, timestamp)
        self.late = True if late == 'true' else False


class EventSubmissionUpdated(Event):
    def __init__(self, student_id, timestamp, late: str):
        assert late in ['true', 'false']
        super().__init__(student_id, timestamp)
        self.late = True if late == 'true' else False


class EventSubmissionCommentCreated(Event):
    def __init__(self, student_id, timestamp, length: int):
        assert length > 0
        super().__init__(student_id, timestamp)
        self.length = length


class EventGradeChange(Event):
    def __init__(self, student_id, timestamp, grade: float):
        assert 0 <= grade <= 1
        super().__init__(student_id, timestamp)
        self.grade = grade


# ===== DATA CLEANING AND PREPROCESSING =====

def extract_event_from_csv_row(row):
    """
    Convert a row from a raw Canvas database CSV file to an Event object.
    Note: row[0] is an unused full user ID, and row[2] is course_id.
    :param row: The line read by csv_reader.
    :return Event
    """
    student_id = int(row[-1])
    timestamp = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S.%f')
    event_name = row[3]
    if event_name == 'asset_accessed':
        return EventAssetAccessed(student_id, timestamp, int(row[4]), row[5], row[6])
    elif event_name == 'attachment_created':
        return EventAttachmentCreated(student_id, timestamp, row[7])
    elif event_name == 'attachment_updated':
        return EventAttachmentUpdated(student_id, timestamp, row[7])
    elif event_name == 'discussion_entry_created' or event_name == 'discussion_entry_submitted':
        if not row[10]:  # post text
            return None
        if not row[9]:  # no parent_entry ID, the discussion post is the first in a thread
            row[9] = 0
        return EventDiscussionEntryCreated(student_id, timestamp, int(row[8]), int(row[9]), len(row[10]))
    elif event_name == 'logged_in':
        return EventLoggedIn(student_id, timestamp)
    elif event_name == 'quiz_submitted':
        return EventQuizSubmitted(student_id, timestamp)
    elif event_name == 'submission_created':
        return EventSubmissionCreated(student_id, timestamp, row[11].lower())
    elif event_name == 'submission_updated':
        return EventSubmissionUpdated(student_id, timestamp, row[11].lower())
    elif event_name == 'submission_comment_created':
        if not row[12]:  # post text
            return None
        return EventSubmissionCommentCreated(student_id, timestamp, len(row[12]))
    elif event_name == 'grade_change':
        if not row[13] or not row[14]:
            return None
        points_awarded = float(row[13])
        points_possible = float(row[14])
        if points_possible == 0:
            return None
        if points_awarded > points_possible:
            points_awarded = points_possible
        grade = points_awarded / points_possible
        return EventGradeChange(student_id, timestamp, grade)
    else:
        pass
        # print('Unknown event name:', event_name)


def preprocess_data(input_filename='data_raw.csv', output_filename='data_preprocessed.csv'):
    """
    Convert the raw input data set exported from the Canvas database to a CSV that can be
    directly used in regression/classification algorithms.
    :param input_filename: Path to the input file with the raw data from Canvas database.
    :param output_filename: Path to the output file with the data ready for regression analysis.
    """
    with open(output_filename, 'w', newline='', encoding='utf-8') as output_csv_file:
        csv_writer = csv.writer(output_csv_file)
        csv_writer.writerow(FEATURE_NAMES)  # First row (header)
        with open(input_filename, encoding='utf-8') as input_csv_file:
            csv_reader = csv.reader(input_csv_file)
            next(csv_reader)  # Skip the first row (header)

            student_id, event_sequence = None, defaultdict(lambda: [])
            for row in csv_reader:
                event = extract_event_from_csv_row(row)
                if event:
                    course_id = int(row[2][13:])  # Up to the 14th digit it's always '1658200000000'
                    if student_id != event.student_id:  # The raw data are sorted by student_id, so reset the calculation
                        student_id = event.student_id
                        event_sequence = defaultdict(lambda: [])
                    event_sequence[course_id].append(event)
                    if isinstance(event, EventGradeChange):  # Final event in sequence
                        write_dataset_row(event_sequence[course_id], course_id, csv_writer)
                        event_sequence[course_id] = []  # Empty the buffer


# ===== DEFINITION OF GENERIC FUNCTIONS FOR COMPUTING FEATURES =====

def get_events_of_type(list_event, event_type):
    return [event for event in list_event if type(event) is event_type]


def count_unique_events(list_event, key_attribute):
    """
    Number of unique events (based on a certain attribute) of a specific type.
    :return int from 0 to infinity
    """
    return len(set(getattr(event, key_attribute) for event in list_event))


def count_events_with_certain_value(list_event, key_attribute, key_value):
    """
    Number of unique events (based on a certain attribute) of a specific type.
    :return int from 0 to infinity
    """
    return len([1 for event in list_event if getattr(event, key_attribute) == key_value])


def time_gaps_between_events(list_event):
    """
    Average and stdev time gaps between events of a specific type.
    :return: floats from almost 0 to infinity, or 0 if list_event is empty
    """
    if len(list_event) <= 1:
        return 0, 0
    posix_timestamps = [event.timestamp.timestamp() for event in list_event]
    return mean(posix_timestamps), stdev(posix_timestamps)


# ===== DEFINITION OF FUNCTIONS FOR COMPUTING SPECIFIC FEATURES =====

def count_assets_accessed_per_active_day(list_event_asset_accessed):
    """
    Number of assets the student accessed per day of actively using the system.
    :return float from 0 to infinity
    """
    assets_accessed = len(list_event_asset_accessed)
    active_days = len(set(event.timestamp.day for event in list_event_asset_accessed))
    return 0 if active_days == 0 else assets_accessed / active_days


def count_discussion_posts(list_event_post, min_length=100):
    """
    Number of student comments with length at least equal to min_length.
    :return int from 0 to infinity
    """
    return len([1 for post in list_event_post if post.length >= min_length])


def percentage_non_late_submissions(list_event_submission):
    """
    Percentage of submissions created/updated that were not late.
    :return float from 0 to 1, or -1 if the given list is empty
    """
    non_late_submissions = count_events_with_certain_value(list_event_submission, 'late', False)
    all_submissions = len(list_event_submission)
    return -1 if all_submissions == 0 else non_late_submissions / all_submissions


# ===== COMPUTE FEATURE VALUES =====

FEATURE_NAMES = [
    'student_id',
    'course_id',

    # Group 1: Asset
    'assets_accessed',
    'unique_assets_accessed',
    'unique_asset_types_accessed',
    'unique_asset_categories_accessed',
    'assets_accessed_per_active_day',
    'avg_asset_access_gap',
    'stdev_asset_access_gap',

    # Group 2: Attachment
    'attachments_created',
    'avg_attachments_created_gap',
    'stdev_attachments_created_gap',
    'attachments_updated',
    # 'avg_attachments_updated_gap',
    # 'stdev_attachments_updated_gap',

    # Group 3: Discussion
    'discussion_posts',
    'new_threads_started',

    # Group 4: Login
    # 'logins_made',
    # 'avg_login_gap',
    # 'stdev_login_gap',

    # Group 5: Quiz
    'quizzes_submitted',
    'avg_quiz_submitted_gap',
    'stdev_quiz_submitted_gap',

    # Group 6: Submission
    'submissions_created',
    'submissions_created_on_time',
    'submissions_created_on_time_ratio',
    'avg_submissions_created_gap',
    'stdev_submissions_created_gap',
    'submissions_updated',
    'submissions_updated_on_time',
    'submissions_updated_on_time_ratio',
    'avg_submissions_updated_gap',
    'stdev_submissions_updated_gap',
    'submission_comments',

    'grade'
]


def compute_all_features(event_sequence):
    features = []

    # Group 1: Asset
    list_event_asset_accessed = get_events_of_type(event_sequence, EventAssetAccessed)
    features.append(len(list_event_asset_accessed))
    features.append(count_unique_events(list_event_asset_accessed, 'asset_id'))
    features.append(count_unique_events(list_event_asset_accessed, 'asset_type'))
    features.append(count_unique_events(list_event_asset_accessed, 'asset_category'))
    features.append(count_assets_accessed_per_active_day(list_event_asset_accessed))
    features.extend(time_gaps_between_events(list_event_asset_accessed))

    # Group 2: Attachment
    list_event_attachment_created = get_events_of_type(event_sequence, EventAttachmentCreated)
    features.append(len(list_event_attachment_created))
    features.extend(time_gaps_between_events(list_event_attachment_created))
    list_event_attachment_updated = get_events_of_type(event_sequence, EventAttachmentUpdated)
    features.append(len(list_event_attachment_updated))
    # features.extend(time_gaps_between_events(list_event_attachment_updated))  # Tested, 0 values

    # Group 3: Discussion
    list_event_discussion = get_events_of_type(event_sequence, EventDiscussionEntryCreated)
    features.append(count_discussion_posts(list_event_discussion))
    features.append(count_events_with_certain_value(list_event_discussion, 'parent_entry_id', 0))

    # Group 4: Login
    # list_event_logged_in = get_events_of_type(event_sequence, EventLoggedIn)
    # features.append(count_events(list_event_logged_in))  # Tested, 0 values
    # features.extend(time_gaps_between_events(list_event_logged_in))  # Tested, 0 values

    # Group 5: Quiz
    list_event_quiz_submitted = get_events_of_type(event_sequence, EventQuizSubmitted)
    features.append(len(list_event_quiz_submitted))
    features.extend(time_gaps_between_events(list_event_quiz_submitted))

    # Group 6: Submission
    list_event_submission_created = get_events_of_type(event_sequence, EventSubmissionCreated)
    features.append(len(list_event_submission_created))
    features.append(count_events_with_certain_value(list_event_submission_created, 'late', False))
    features.append(percentage_non_late_submissions(list_event_submission_created))
    features.extend(time_gaps_between_events(list_event_submission_created))
    list_event_submission_updated = get_events_of_type(event_sequence, EventSubmissionUpdated)
    features.append(len(list_event_submission_updated))
    features.append(count_events_with_certain_value(list_event_submission_created, 'late', False))
    features.append(percentage_non_late_submissions(list_event_submission_updated))
    features.extend(time_gaps_between_events(list_event_submission_updated))
    list_event_comment = get_events_of_type(event_sequence, EventSubmissionCommentCreated)
    features.append(count_discussion_posts(list_event_comment))

    assert len(features) == len(FEATURE_NAMES) - 3  # student_id, course_id, grade
    return features


def write_dataset_row(event_sequence, course_id, csv_writer):
    """
    Compute features from the given event_sequence and write them out to CSV.
    :param event_sequence: A list of Events (possibly of mixed type).
    :param course_id: Numerical ID to describe the course context.
    :param csv_writer: An object handling the file output.
    :return: None
    """
    student_id = event_sequence[0].student_id
    # All events in the sequence must belong to the same student
    assert all(event.student_id == student_id for event in event_sequence)

    features = compute_all_features(event_sequence)
    grade = event_sequence[-1].grade  # Target variable
    csv_writer.writerow([student_id, course_id, *features, grade])


# ===== EXAMINE THE FEATURE VALUES AFTER THE COMPUTATION OF DATA_PREPROCESSED.CSV =====

def export_unique_student_ids(df, output_filename='ids-final.txt'):
    student_ids = df['student_id']
    unique_student_ids = pd.Series(student_ids.unique())
    unique_student_ids.to_csv(output_filename, sep=' ', index=False, header=False)


def compute_descriptive_statistics(df):
    df.describe().to_csv('feature_description.csv')
    grades = df['grade']
    mean_grade = grades.mean()
    print('Median grade:', grades.median())
    print('Average grade:', mean_grade)
    print('Total number of grade entries (rows):', grades.count())
    print('Number of grade entries above average grade:', grades[grades > mean_grade].count())


def compute_feature_correlation(df):
    corr_matrix = df.corr(method="spearman")
    plt.figure(figsize=(20, 10), facecolor="white", layout='compressed')
    sns.heatmap(
        corr_matrix,
        vmin=-1, vmax=1,  # because correlation coefficients
        annot=True,
        annot_kws={"fontsize": 8},
        fmt=".2f",
        cmap="vlag"
    )
    plt.savefig('feature_correlation_matrix.png')


def postprocess_data(input_filename='data_preprocessed.csv'):
    df = pd.read_csv(input_filename)
    export_unique_student_ids(df)
    compute_descriptive_statistics(df)
    compute_feature_correlation(df)


# ===== MAIN =====

if __name__ == '__main__':
    preprocess_data()
    postprocess_data()
