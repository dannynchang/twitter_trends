"""Visualizing Twitter Sentiment Across America"""

import string
from data import word_sentiments, load_tweets
from geo import us_states, geo_distance, make_position, longitude, latitude
from maps import draw_state, draw_name, draw_dot, wait
from ucb import main, trace, interact, log_current_line


# Phase 1: The feelings in tweets

def make_tweet(text, time, lat, lon):
   """Return a tweet, represented as a python dictionary.

   text -- A string; the text of the tweet, all in lowercase
   time -- A datetime object; the time that the tweet was posted
   lat -- A number; the latitude of the tweet's location
   lon -- A number; the longitude of the tweet's location
   """
   return {'text': text, 'time': time, 'latitude': lat, 'longitude': lon}

def tweet_words(tweet):
   """Return a list of words in the tweet.

   Arguments:
   tweet -- a tweet abstract data type.

   Return 1 value:
    - The list of words in the tweet.
   """
   return tweet['text']

def tweet_location(tweet):
   """Return a position (see geo.py) that represents the tweet's
location."""
   return tweet['latitude'], tweet['longitude']

def tweet_string(tweet):
   """Return a string representing the tweet."""
   return '"{0}" @ {1}'.format(tweet['text'], tweet_location(tweet))

def extract_words(text):
   """Return the words in a tweet, not including punctuation.

   >>> extract_words('anything else.....not my job')
   ['anything', 'else', 'not', 'my', 'job']
   >>> extract_words('i love my job. #winning')
   ['i', 'love', 'my', 'job', 'winning']
   >>> extract_words('make justin # 1 by tweeting #vma #justinbieber :)')
   ['make', 'justin', 'by', 'tweeting', 'vma', 'justinbieber']
   >>> extract_words("paperclips! they're so awesome, cool, & useful!")
   ['paperclips', 'they', 're', 'so', 'awesome', 'cool', 'useful']
   """
   for word in text:
       if word == '.' or word == '\'':
           b = text.replace(word, ' ')
           text = b
       elif word not in string.ascii_letters and word != ' ':
           b = text.replace(word, '')
           text = b
   return text.split()

def get_word_sentiment(word):
   """Return a number between -1 and +1 representing the degree of
positive or
   negative feeling in the given word.

   Return None if the word is not in the sentiment dictionary.
   (0 represents a neutral feeling, not an unknown feeling.)

   >>> get_word_sentiment('good')
   0.875
   >>> get_word_sentiment('bad')
   -0.625
   >>> get_word_sentiment('winning')
   0.5
   >>> get_word_sentiment('Berkeley')  # Returns None
   """
   return word_sentiments.get(word, None)

def analyze_tweet_sentiment(tweet):
   """ Return a number between -1 and +1 representing the degree of positive or
   negative sentiment in the given tweet, averaging over all the words in the
   tweet that have a sentiment score.

   If there are words that don't have a sentiment score, leave them
   out of the calculation.

   If no words in the tweet have a sentiment score, return None.
   (do not return 0, which represents neutral sentiment).

   >>> positive = make_tweet('i love my job. #winning', None, 0, 0)
   >>> round(analyze_tweet_sentiment(positive), 5)
   0.29167
   >>> negative = make_tweet("Thinking, 'I hate my job'", None, 0, 0)
   >>> analyze_tweet_sentiment(negative)
   -0.25
   >>> no_sentiment = make_tweet("Go bears!", None, 0, 0)
   >>> analyze_tweet_sentiment(no_sentiment)
   """
   average = None
   score_sum = 0
   string = extract_words(tweet)
   for words in string:
      score_sum += get_word_sentiment(words)
   average = score_sum / len(string)
   return average

@main
def print_sentiment(text='Are you virtuous or verminous?'):
   """Print the words in text, annotated by their sentiment scores.

   For example, to print each word of a sentence with its sentiment:

   # python3 trends.py "computer science is my favorite!"
   """
   words = extract_words(text.lower())
   assert words, 'No words extracted from "' + text + '"'
   layout = '{0:>' + str(len(max(words, key=len))) + '}: {1}'
   for word in extract_words(text.lower()):
     print(layout.format(word, get_word_sentiment(word)))
      
# Phase 2: The geometry of maps

def find_centroid(polygon):
    """Find the centroid of a polygon.

    http://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
    
    polygon -- A list of positions, in which the first and last are the same

    Returns: 3 numbers; centroid latitude, centroid longitude, and polygon area

    Hint: If a polygon has 0 area, return its first position as its centroid

    >>> p1, p2, p3 = make_position(1, 2), make_position(3, 4), make_position(5, 0)
    >>> triangle = [p1, p2, p3, p1]  # First vertex is also the last vertex
    >>> find_centroid(triangle)
    (3.0, 2.0, 6.0)
    >>> find_centroid([p1, p2, p1])
    (1, 2, 0)
    """
   centroid = list[0, 0, 0]6x
   signed_area = 0.0
   cvs = node.getCVs(space='world')
   v0 = cvs[len(cvs) - 1]
   for i, cv in enumerate(cvs[:-1]):
      v1 = cv
      a = v0.x * v1.y - v1.x * v0.y
      signed_area += a
      centroid += sum([v0, v1]) * a
      v0 = v1
   signed_area *= 0.5
   centroid /= 6 * signed_area
   return centroid
    

def find_center(shapes):
    """Compute the geographic center of a state, averaged over its shapes.

    The center is the average position of centroids of the polygons in shapes,
    weighted by the area of those polygons.
    
    Arguments:
    shapes -- a list of lists of polygons

    >>> ca = find_center(us_states['CA'])  # California
    >>> round(latitude(ca), 5)
    37.25389
    >>> round(longitude(ca), 5)
    -119.61439

    >>> hi = find_center(us_states['HI'])  # Hawaii
    >>> round(latitude(hi), 5)
    20.1489
    >>> round(longitude(hi), 5)
    -156.21763
    """
   centroid = ()
   get_lat, get_lon, get_area = centroid[0], centroid[1], centroid[2]
   for polygon in shapes:
      centroid = centroid + find_centroid(polygon)
   help_lat, help_lon, area_sum = 0, 0, 0  #Variables used in calculating the centers
   for item in centroid:
      area_sum = area_sum + get_area(item)
      help_lat = help_lat + (get_lat(item) * get_area(item))
      help_lon = help_lon + (get_lon(item) * get_area(item))
   lat = (help_lat / area_sum)
   lon = (help_lon / area_sum)
   return make_location(lat, lon)
      
      


# Uncomment this decorator during Phase 2.
# @main
def draw_centered_map(center_state='TX', n=10):
    """Draw the n states closest to center_state.
    
    For example, to draw the 20 states closest to California (including California):

    # python3 trends.py CA 20
    """
    us_centers = {n: find_center(s) for n, s in us_states.items()}
    center = us_centers[center_state.upper()]
    dist_from_center = lambda name: geo_distance(center, us_centers[name])
    for name in sorted(us_states.keys(), key=dist_from_center)[:int(n)]:
      draw_state(us_states[name])
      draw_name(name, us_centers[name])
    draw_dot(center, 1, 10)  # Mark the center state with a red dot
    wait()


# Phase 3: The mood of the nation

def find_closest_state(tweet, state_centers):
    """Return the name of the state closest to the given tweet's location.
    
    Use the geo_distance function (already provided) to calculate distance 
    in miles between two latitude-longitude positions.

    Arguments:
    tweet -- a tweet abstract data type
    state_centers -- a dictionary from state names to state shapes

    >>> us_centers = {n: find_center(s) for n, s in us_states.items()}
    >>> sf = make_tweet("Welcome to San Francisco", None, 38, -122)
    >>> ny = make_tweet("Welcome to New York", None, 41, -74)
    >>> find_closest_state(sf, us_centers)
    'CA'
    >>> find_closest_state(ny, us_centers)
    'NJ'
    """
    location = tweet_location(tweet)
   state_info = {'state': None, 'distance': 0}
   for state in state_centers:
      if state_info['distance'] > geo_distance(location, state):
         state_info['state'] = state
         state_info['distance'] = geo_distance(location, state)
   return state_info['state']
   

def group_tweets_by_state(tweets):
    """Return a dictionary that aggregates tweets by their nearest state center.

    The keys of the returned dictionary are state names, and the values are
    lists of tweets that appear closer to that state center than any other.
    
    tweets -- a sequence of tweet abstract data types    

    >>> sf = make_tweet("Welcome to San Francisco", None, 38, -122)
    >>> ny = make_tweet("Welcome to New York", None, 41, -74)
    >>> ca_tweets = group_tweets_by_state([sf, ny])['CA']
    >>> tweet_string(ca_tweets[0])
    '"Welcome to San Francisco" @ (38, -122)'
    """
    tweets_by_state = {}
    for item in tweets:
      state = find_closest_state(item, state_centers)
      tweet = tweet_words(item)
      if state not in tweets_by_state:
         tweets_by_state[state] = tweet
      else:
         tweets_by_state[state].append(item)
    return tweets_by_state

def calculate_average_sentiments(tweets_by_state):
    """Calculate the average sentiment of the states by averaging over all 
    the tweets from each state. Return the result as a dictionary from state
    names to average sentiment values.
   
    If a state has no tweets with sentiment values, leave it out of the
    dictionary entirely.  Do not include a states with no tweets, or with tweets
    that have no sentiment, as 0.  0 represents neutral sentiment, not unknown
    sentiment.

    tweets_by_state -- A dictionary from state names to lists of tweets
    """
    averaged_state_sentiments = {}
   for tweet_state in tweets_by_state:
      sentiment_score = 0
      for tweet in tweet_state:
         sentiment_score = analyze_tweet_sentiment(tweet) + sentiment_score
         if sentiment_score != None:
            averaged_state_sentiment[tweet_state] = sentiment+score
         else:
            return None
   return averaged_state_sentiments

def draw_state_sentiments(state_sentiments={}):
    """Draw all U.S. states in colors corresponding to their sentiment value.
    
    Unknown state names are ignored; states without values are colored grey.
    
    state_sentiments -- A dictionary from state strings to sentiment values
    """
    for name, shapes in us_states.items():
      sentiment = state_sentiments.get(name, None)
      draw_state(shapes, sentiment)
    for name, shapes in us_states.items():
      center = find_center(shapes)
      if center is not None:
        draw_name(name, center)

# Uncomment this decorator during Phase 3.
# @main
def draw_map_for_term(term='my job'):
    """
    Draw the sentiment map corresponding to the tweets that match term.
    
    term -- a word or phrase to filter the tweets by.  
    
    To visualize tweets containing the word "obama":
    
    # python3 trends.py obama
    
    Some term suggestions:
    New York, Texas, sandwich, my life, justinbieber
    """
    tweets = load_tweets(make_tweet, term)
    tweets_by_state = group_tweets_by_state(tweets)
    state_sentiments = calculate_average_sentiments(tweets_by_state)
    draw_state_sentiments(state_sentiments)
    for tweet in tweets:
      draw_dot(tweet_location(tweet), analyze_tweet_sentiment(tweet))
    wait()


"""Geography and projection utilities."""

from math import sin, cos, atan2, radians, sqrt
from json import JSONDecoder

def make_position(lat, lon):
    """Return a geographic position, which has a latitude and longitude."""
    return (lat, lon)

def latitude(position):
    """Return the latitudinal coordinate of a geographic position."""
    return position[0]

def longitude(position):
    """Return the longitudinal coordinate of a geographic position."""
    return position[1]

def geo_distance(position1, position2):
    """Return the great circle distance (in miles) between two 
    geographic positions.
    
    Uses the "haversine" formula.
    http://en.wikipedia.org/wiki/Haversine_formula

    >>> round(geo_distance(make_position(50, 5), make_position(58, 3)), 1)
    559.2
    """
    earth_radius = 3963.2  # miles
    lat1, lat2 = [radians(latitude(p)) for p in (position1, position2)]
    lon1, lon2 = [radians(longitude(p)) for p in (position1, position2)]
    dlat, dlon = lat2-lat1, lon2-lon1
    a = sin(dlat/2) ** 2  + sin(dlon/2) ** 2 * cos(lat1) * cos(lat2)
    c = 2 * atan2(sqrt(a), sqrt(1-a)); 
    return earth_radius * c;

def position_to_xy(position):
    """Convert a geographic position within the US to a planar x-y point."""
    lat = latitude(position)
    lon = longitude(position)
    if lat < 25:
      return _hawaii(position)
    elif lat > 52:
      return _alaska(position)
    else:
      return _lower48(position)

def albers_projection(origin, parallels, translate, scale):
    """Return an Albers projection from geographic positions to x-y positions.
    
    Derived from Mike Bostock's Albers javascript implementation for D3
    http://mbostock.github.com/d3
    http://mathworld.wolfram.com/AlbersEqual-AreaConicProjection.html

    origin -- a geographic position
    parallels -- bounding latitudes
    translate -- x-y translation to place the projection within a larger map
    scale -- scaling factor
    """
    phi1, phi2 = [radians(p) for p in parallels]
    base_lat = radians(latitude(origin))
    s, c = sin(phi1), cos(phi1)
    base_lon = radians(longitude(origin))
    n = 0.5 * (s + sin(phi2))
    C = c*c + 2*n*s
    p0 = sqrt(C - 2*n*sin(base_lat))/n

    def project(position):
      lat, lon = radians(latitude(position)), radians(longitude(position))
      t = n * (lon - base_lon)
      p = sqrt(C - 2*n*sin(lat))/n
      x = scale * p * sin(t) + translate[0]
      y = scale * (p * cos(t) - p0) + translate[1]
      return (x, y)
    return project

_lower48 = albers_projection(make_position(38, -98), [29.5, 45.5], [480,250], 1000)
_alaska = albers_projection(make_position(60, -160), [55,65], [150,440], 400)
_hawaii = albers_projection(make_position(20, -160), [8,18], [300,450], 1000)

def load_states():
    """Load the coordinates of all the state outlines and return them 
    in a dictionary, from names to shapes lists.

    >>> len(load_states()['HI'])  # Hawaii has 5 islands
    5
    """
    json_data_file = open("data/states.json", encoding='utf8')
    states = JSONDecoder().decode(json_data_file.read())
    for state, shapes in states.items():
      for index, shape in enumerate(shapes):
        if type(shape[0][0]) == list:  # the shape is a single polygon
          assert len(shape) == 1, 'Multi-polygon shape'
          shape = shape[0]
        shapes[index] = [make_position(*reversed(pos)) for pos in shape]
    return states

us_states = load_states()
