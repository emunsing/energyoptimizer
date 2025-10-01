

def test_shift_copy_dataset_to_new_index():
    """
    expected failing input cases:
        - Less-than-full-year (2015-01-01 to 2015-12-31 00:00; raise error);
        - 1-year hourly input data but missing hours
        - timezone unaware input_dataset
        - timezone unaware new_time_index

    Input Cases:
    - 15-minute vs. 30-min vs. 1-hr input
    - Leap-year vs non-leap-year input
    - Input span: 1-year on-year-end (jan 1 - Dec 31), 1-year off-year-end (e.g. Jun 1- May 31), multi-year on-year-end (2015-01-01:2016-12-31), multi-yaer off-year-end (2015-06-01:2017-05-31)

    Target cases: Assume 3-min target frequency
    - 1-year, non-overlapping, non-leap year
    - 1-year, non-overlapping, leap-year
    - 1-year, part-overlapping with input dataset, non-leap-year
    - 1-year, part-overlapping with input dataset, leap-year
    - partial-year, non-overlapping, non-leap-year
    - 1-and-a-half-year, non-overlapping, non-leap-year
    - multi-year, non-overlapping
    - multi-year, part-overlapping with input dataset

    Gotchas:
    - Leap-year
    - DST shifts
    - Ensure that multi-year input data is
    - Want to be able to utilize information from all years of the input data
    -

    :return:
    """
