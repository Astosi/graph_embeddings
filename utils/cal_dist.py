# Calculate the distance between pairs
def calculate_dist(point_pairs):
    """
    To set the thresholds I did some tests.

    for latitude change:
    dist( {lat: 52.08, long: 4.97} , {lat: 52.09, long: 4.97} ) == 1.11 km
    dist( {lat: 52.08, long: 4.97} , {lat: 52.10, long: 4.97} ) == 2.23 km

    for longtitude change:
    dist( {lat: 52.08, long: 4.97} , {lat: 52.08, long: 4.98} ) == 0.68 km
    dist( {lat: 52.08, long: 4.97} , {lat: 52.08, long: 4.99} ) == 1.37 km
  """

    try:
        p1 = point_pairs[0]
        p2 = point_pairs[1]
    except IndexError:
        # Handle the case where point_pairs is not a list of two points
        print("IndexError!!")
        return 0

    # Set the thresholds based on your tests
    # lat_threshold = 0.02
    # lon_threshold = 0.03

    lat_threshold = 0.03
    lon_threshold = 0.04

    lat1, lat2, lon1, lon2 = p1.y, p2.y, p1.x, p2.x

    # # Check if the distance is large enough to warrant calculating the distance
    # if (abs(lon1 - lon2) > lon_threshold or abs(lat1 - lat2) > lat_threshold):
    #     return 0

    from geopy import distance

    # Calculate the distance using the Vincenty formula
    distance = round(distance.great_circle((lat1, lon1), (lat2, lon2)).km, 1)

    return distance
