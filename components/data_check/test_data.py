'''
Author: Vitor Abdo

This .py file runs the necessary tests to check our data 
after cleaning it after the "basic_clean" step
'''

def test_column_names(data):
    '''Tests if the column names are the same as the original 
    file, including in the same order
    '''
    expected_colums = [
       'id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
       'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
       'minimum_nights', 'number_of_reviews', 'last_review',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365', 'number_of_reviews_ltm', 'license']

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)