"""Visualizing Twitter Sentiment Across America"""

import string
import re

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
    >>make_tweet('herp the derp.', '1', '2', '3')
    {'herp', 'the', 'derp.'}
    """
    return tweet['text'].split(' ')

def tweet_location(tweet):
    """Return a position (see geo.py) that represents the tweet's location."""
    return make_position(tweet['latitude'], tweet['longitude'])

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
    stripped_text = re.sub('[^A-Za-z]+', ' ', text)
    return stripped_text.split()

def get_word_sentiment(word):
    """Return a number between -1 and +1 representing the degree of positive or
    negative feeling in the given word. 

    Return None if the word is not  in the sentiment dictionary.
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
    total, count = 0, 0
    
    for each in extract_words(tweet['text']):
        if get_word_sentiment(each) == None:
            pass
        else:
            total += get_word_sentiment(each)
            count += 1
    if total != 0:
        return total / count
    
# @main
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
    >>> find_centroid([p1, p3, p2, p1])
    (3.0, 2.0, 6.0)
    >>> find_centroid([p1, p2, p1])
    (1, 2, 0)
    """
    def area(polygon):
        area=0
        total_points = len(polygon)
        j=total_points-1
        i = 0
        for _ in polygon:
            p1=polygon[i]
            p2=polygon[j]
            area += (p1[0]*p2[1]) - (p1[1]*p2[0])
            j=i
            i+=1
        return area/2
    
    def centroid(polygon):
        if area(polygon) == 0:
            return polygon[0][0], polygon[0][1], 0
        else:
            total_points = len(polygon)
            x, y, i, j = 0, 0, 0, total_points-1
            for _ in polygon:
                p1, p2 = polygon[i], polygon[j]
                f = p1[0] * p2[1] - p2[0] * p1[1]
                x += (p1[0]+p2[0]) * f
                y += (p1[1]+p2[1]) * f
                j = i
                i += 1
            f=area(polygon)*6
            return x/f, y/f, abs(area(polygon))
    return centroid(polygon)
    
    

def find_center(shapes):
    """Compute the geographic center of a state, averaged over its shapes.

    The center is the average position of centroids of the polygons in shapes,
    weighted by the area of those polygons.
    
    Arguments:
    shapes -- a list of polygons

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
    total_x, total_y, total_area = 0, 0, 0
    for polygon in shapes:
        result = find_centroid(polygon)
        total_x += result[0] * result[2]
        total_y += result[1] * result[2]
        total_area += result[2]
    total_shapes = len(shapes)
    return total_x/(total_area), total_y/(total_area)


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
    least = {'dist': 30000, 'name':None}
    for state in state_centers:
        tweet_pos = make_position(tweet['latitude'], tweet['longitude'])
        center_pos = state_centers[state]
        dist = geo_distance(tweet_pos, center_pos)
        if dist < least['dist']:
            least['dist'] = dist
            least['name'] = state
    return least['name']

def group_tweets_by_state(tweets):
    """Return a dictionary that aggregates tweets by their nearest state center.

    The keys of the returned dictionary are state names, and the values are
    lists of tweets that appear closer to that state center than any other.
    
    tweets -- a sequence of tweet abstract data types    

    >>> sf = make_tweet("Welcome to San Francisco", None, 38, -122)
    >>> ny = make_tweet("Welcome to New York", None, 41, -74)
    >>> sf2 = make_tweet("Welcome to San Francisco 2", None, 38, -122)
    >>> tweet_list = group_tweets_by_state([sf, ny, sf2])
    >>> ca_tweets = group_tweets_by_state([sf, ny, sf2])['CA']
    >>> tweet_string(ca_tweets[0])
    '"Welcome to San Francisco" @ (38, -122)'
    """
    us_centers = {n: find_center(s) for n, s in us_states.items()}
    tweets_by_state = {}
    for tweet in tweets:
        tweet_location = find_closest_state(tweet, us_centers)
        if tweet_location not in tweets_by_state:
            tweets_by_state[tweet_location] = [tweet]
        else:
            tweets_by_state[tweet_location].append(tweet)
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
    for state in tweets_by_state:
        total = 0
        for tweet in tweets_by_state[state]:
            print(tweet)
            sentiment = analyze_tweet_sentiment(tweet)
            if sentiment != None:
                total += sentiment
        if total != 0:
            averaged_state_sentiments[state] = total/len(state)
    print(averaged_state_sentiments)
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
#@main
def draw_map_for_term(term='test search obama'):
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

	
	
	
	
"""Functions for reading data from the sentiment dictionary and tweet files."""

import os
import re
import string
from datetime import datetime
from ucb import main, interact

DATA_PATH = 'data' + os.sep

def load_sentiments(file_name="data"+os.sep+"sentiments.csv"):
    """Read the sentiment file and return a dictionary containing the sentiment
    score of each word, a value from -1 to +1.
    """
    sentiments = {}
    for line in open(file_name, encoding='utf8'):
        word, score = line.split(',')
        sentiments[word] = float(score.strip())
    return sentiments

word_sentiments = load_sentiments()

def file_name_for_term(term):
    """Return a valid filename that corresponds to an arbitrary term string."""
    valid_characters = '-_' + string.ascii_letters + string.digits
    no_space = term.replace(' ', '_')
    return ''.join(c for c in no_space if c in valid_characters) + '.txt'

def generate_filtered_file(unfiltered_name, term):
    """Return the path to a file containing tweets that match term, generating
    that file if necessary.
    """
    filtered_path = DATA_PATH + file_name_for_term(term)
    if not os.path.exists(filtered_path):
        print('Generating filtered tweets file for "{0}".'.format(term))
        r = re.compile('\W' + term + '\W', flags=re.IGNORECASE)
        with open(filtered_path, mode='w', encoding='utf8') as out:
            unfiltered = open(DATA_PATH + unfiltered_name, encoding='utf8')
            matches = [l for l in unfiltered if term in l.lower()]
            for line in matches:
                if r.search(line):
                    out.write(line)
    return filtered_path

def load_tweets(make_tweet, term='my job', file_name='all_tweets.txt'):
    """Return the list of tweets in file_name that contain term.
    
    make_tweet -- a constructor that takes four arguments:
      - a string containing the words in the tweet
      - a datetime.datetime object representing the time of the tweet
      - a longitude coordinate
      - a latitude coordinate
    """
    term = term.lower()
    filtered_path = generate_filtered_file(file_name, term)
    tweets = []
    for line in open(filtered_path, encoding='utf8'):
        if len(line.strip().split("\t")) >=4:
            loc, _, time_text, text = line.strip().split("\t")
            time = datetime.strptime(time_text, '%Y-%m-%d %H:%M:%S')
            lat, lon = eval(loc)
            tweet = make_tweet(text.lower(), time, lat, lon)
            tweets.append(tweet)
    return tweets

	
	
	
	
	
	
	
	
	
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












"""The graphics module implements a simple GUI library."""

import sys
import math

try:
    import tkinter
except Exception(e):
    print('Could not load tkinter: ' + str(e))

FRAME_TIME = 1/30

class Canvas(object):
    """A Canvas object supports drawing and animation primitives.

    draw_* methods return the id number of a shape object in the underlying Tk
    object.  This id can be passed to move_* and edit_* methods.
    
    Canvas is a singleton; only one Canvas instance can be created.

    """

    _instance = None

    def __init__(self, width=1024, height=768, title='', color='White', tk=None):
        # Singleton enforcement
        if Canvas._instance is not None:
            raise Exception('Only one canvas can be instantiated.')
        Canvas._instance = self

        # Attributes
        self.color = color
        self.width = width
        self.height = height

        # Root window
        self._tk = tk or tkinter.Tk()
        self._tk.protocol('WM_DELETE_WINDOW', sys.exit)
        self._tk.title(title or 'Graphics Window')
        self._tk.bind('<Button-1>', self._click)
        self._click_pos = None
        
        # Canvas object
        self._canvas = tkinter.Canvas(self._tk, width=width, height=height)
        self._canvas.pack()
        self._draw_background()
        self._canvas.update()
        self._images = dict()

    def clear(self, shape='all'):
        """Clear all shapes, text, and images."""
        self._canvas.delete(shape)
        if shape == 'all':
            self._draw_background()
        self._canvas.update()

    def draw_polygon(self, points, color='Black', fill_color=None, filled=1, smooth=0, width=1):
        """Draw a polygon and return its tkinter id.

        points -- a list of (x, y) pairs encoding pixel positions
        """
        if fill_color == None: 
            fill_color = color
        if filled == 0: 
            fill_color = ""
        return self._canvas.create_polygon(flattened(points), outline=color, fill=fill_color, 
                smooth=smooth, width=width)

    def draw_circle(self, center, radius, color='Black', fill_color=None, filled=1, width=1):
        """Draw a cirlce and return its tkinter id.

        center -- an (x, y) pair encoding a pixel position
        """
        if fill_color == None: 
            fill_color = color
        if filled == 0: 
            fill_color = ""
        x0, y0 = [c - radius for c in center]
        x1, y1 = [c + radius for c in center]
        return self._canvas.create_oval(x0, y0, x1, y1, outline=color, fill=fill_color, width=width)

    def draw_image(self, pos, image_file=None, scale=1, anchor=tkinter.NW):
        """Draw an image from a file and return its tkinter id."""
        key = (image_file, scale)
        if key not in self._images:
            image = tkinter.PhotoImage(file=image_file)
            if scale >= 1:
                image = image.zoom(int(scale))
            else:
                image = image.subsample(int(1/scale))
            self._images[key] = image 

        image = self._images[key]
        x, y = pos
        return self._canvas.create_image(x, y, image=image, anchor=anchor)

    def draw_text(self, text, pos, color='Black', font='Arial', 
                  size=12, style='normal', anchor=tkinter.NW):
        """Draw text and return its tkinter id."""
        x, y = pos
        font = (font, str(size), style)
        return self._canvas.create_text(x, y, fill=color, text=text, font=font, anchor=anchor)

    def edit_text(self, id, text=None, color=None, font=None, size=12,
                  style='normal'):
        """Edit the text, color, or font of an existing text object."""
        if color is not None:
            self._canvas.itemconfigure(id, fill=color)
        if text is not None:
            self._canvas.itemconfigure(id, text=text)
        if font is not None:
            self._canvas.itemconfigure(id, font=(font, str(size), style))

    def animate_shape(self, id, duration, points_fn, frame_count=0):
        """Animate an existing shape over points.""" 
        max_frames = duration // FRAME_TIME
        points = points_fn(frame_count)
        self._canvas.coords(id, flattened(points))
        if frame_count < max_frames:
            def tail():
                """Continues the animation at the next frame."""
                self.animate_shape(id, duration, points_fn, frame_count + 1)
            self._tk.after(int(FRAME_TIME * 1000), tail)

    def slide_shape(self, id, end_pos, duration, elapsed=0):
        """Slide an existing shape to end_pos."""
        points = paired(self._canvas.coords(id))
        start_pos = points[0]
        max_frames = duration // FRAME_TIME
        def points_fn(frame_count):
            completed = frame_count / max_frames
            offset = [(e - s) * completed for s, e in zip(start_pos, end_pos)]
            return [shift_point(p, offset) for p in points]
        self.animate_shape(id, duration, points_fn)

    def wait_for_click(self, seconds=0):
        """Return (position, elapsed) pair of click position and elapsed time.
        
        position: (x,y) pixel position of click
        elapsed:  milliseconds elapsed since call
        seconds:  maximum number of seconds to wait for a click

        If there is still no click after the given time, return (None, seconds).
        
        """
        elapsed = 0
        while elapsed < seconds or seconds == 0:
            if self._click_pos is not None:
                pos = self._click_pos
                self._click_pos = None
                return pos, elapsed
            self._sleep(FRAME_TIME)
            elapsed += FRAME_TIME
        return None, elapsed

    def _draw_background(self):
        w, h = self.width - 1, self.height - 1
        corners = [(0,0), (0, h), (w, h), (w, 0)]
        self.draw_polygon(corners, self.color, fill_color=self.color, filled=True, smooth=False)

    def _click(self, event):
        self._click_pos = (event.x, event.y)

    def _sleep(self, seconds):
        self._tk.update_idletasks()
        self._tk.after(int(1000 * seconds), self._tk.quit)
        self._tk.mainloop()

def flattened(points):
    """Return a flat list of coordinates from a list of pairs."""
    coords = list()
    [coords.extend(p) for p in points]
    return tuple(coords)

def paired(coords):
    """Return a list of pairs from a flat list of coordinates.""" 
    assert len(coords) % 2 == 0, 'Coordinates are not paired.'
    points = []
    x = None
    for elem in coords:
        if x is None:
            x = elem
        else:
            points.append((x, elem))
            x = None
    return points

def translate_point(point, angle, distance):
    """Translate a point a distance in a direction (angle)."""
    x, y = point
    return (x + math.cos(angle) * distance, y + math.sin(angle) * distance)

def shift_point(point, offset):
    """Shift a point by an offset."""
    x, y = point
    dx, dy = offset
    return (x + dx, y + dy)

def rectangle_points(pos, width, height):
    """Return the points of a rectangle starting at pos."""
    x1, y1 = pos
    x2, y2 = width + x1, height + y1
    return [(x1, y1), (x1, y2), (x2, y2), (x2, y1)]

def format_color(r, g, b):
    """Format a color as a string.

    r, g, b -- integers from 0 to 255
    """
    return '#{0:02x}{1:02x}{2:02x}'.format(int(r * 255), int(g * 255), int(b * 255))
"""Map drawing utilities for U.S. sentiment data."""

from graphics import Canvas
from geo import position_to_xy, us_states

# A fixed gradient of sentiment colors from negative (blue) to positive (red)
# Colors chosen via Cynthia Brewer's Color Brewer (colorbrewer2.com)
SENTIMENT_COLORS = ["#313695", "#4575B4", "#74ADD1", "#ABD9E9", "#E0F3F8",
                    "#FFFFFF", "#FEE090", "#FDAE61", "#F46D43", "#D73027",
                    "#A50026"]
GRAY = "#AAAAAA"

def get_sentiment_color(sentiment, sentiment_scale=4):
    """Returns a color corresponding to the sentiment value.

    sentiment -- a number between -1 (negative) and +1 (positive)
    """
    if sentiment is None:
        return GRAY
    scaled = (sentiment_scale * sentiment + 1)/2
    index = int( scaled * len(SENTIMENT_COLORS) ) # Rounds down
    if index < 0:
        index = 0
    if index >= len(SENTIMENT_COLORS):
        index = len(SENTIMENT_COLORS) - 1
    return SENTIMENT_COLORS[index]

def draw_state(shapes, sentiment=None):
    """Draw the named state in the given color on the canvas.
    
    state -- a list of list of polygons (which are lists of positions)
    sentiment -- a number between -1 (negative) and 1 (positive)
    canvas -- the graphics.Canvas object
    """
    for polygon in shapes:
        vertices = [position_to_xy(position) for position in polygon]
        color = get_sentiment_color(sentiment)
        get_canvas().draw_polygon(vertices, fill_color=color)

def draw_name(name, location):
    """Draw the two-letter postal code at the center of the state.
    
    location -- a position
    """
    center = position_to_xy(location)
    get_canvas().draw_text(name.upper(), center, anchor='center', style='bold')

def draw_dot(location, sentiment=None, radius=3):
    """Draw a small dot at location.

    location -- a position
    sentiment -- a number between -1 (negative) and 1 (positive)
    """
    center = position_to_xy(location)
    color = get_sentiment_color(sentiment)
    get_canvas().draw_circle(center, radius, fill_color=color)

def memoize(fn):
    """A decorator for caching the results of the decorated function."""
    cache = {}
    def memoized(*args):
        if args in cache:
            return cache[args]
        result = fn(*args)
        cache[args] = result
        return result
    return memoized

@memoize
def get_canvas():
    """Return a Canvas, which is a drawing window."""
    return Canvas(width=960, height=500)

def wait():
    """Wait for mouse click."""
    get_canvas().wait_for_click()

	
	
	
	
"""The ucb module contains functions specific to 61A at UC Berkeley."""

import code
import functools
import inspect
import re
import sys

        
def main(fn):
    """Call fn with command line arguments.  Used as a decorator.

    The main decorator marks the function that starts a program. For example,
    
    @main
    def my_run_function():
        # function body
    
    Use this instead of the typical __name__ == "__main__" predicate.
    """
    if inspect.stack()[1][0].f_locals['__name__'] == '__main__':
        args = sys.argv[1:] # Discard the script name from command line
        fn(*args) # Call the main function


PREFIX = ''
def trace(fn):
    """A decorator that prints a function's name, its arguments, and its return
    values each time the function is called. For example,

    @trace
    def compute_something(x, y):
        # function body
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwds):
        global PREFIX
        reprs = [repr(e) for e in args] 
        reprs += [repr(k) + '=' + repr(v) for k, v in kwds.items()]
        log('{0}({1})'.format(fn.__name__, ', '.join(reprs)) + ':')
        PREFIX += '    '
        try:
            result = fn(*args, **kwds)
            PREFIX = PREFIX[:-4]
        except Exception as e:
            log(fn.__name__ + ' exited via exception')
            PREFIX = PREFIX[:-4]
            raise
        # Here, print out the return value.
        log('{0}({1}) -> {2}'.format(fn.__name__, ', '.join(reprs), result))
        return result
    return wrapped


def log(message):
    """Print an indented message (used with trace)."""
    if type(message) is not str:
        message = str(message)
    print(PREFIX + re.sub('\n', '\n' + PREFIX, message))


def log_current_line():
    """Print information about the current line of code."""
    frame = inspect.stack()[1]
    log('Current line: File "{f[1]}", line {f[2]}, in {f[3]}'.format(f=frame))


def interact():
    """Start an interactive interpreter session in the current environment.

    On Unix:
      <Control>-D exits the interactive session and returns to normal execution.
    In Windows:
      <Control>-z <Enter> exists the interactive session and returns to normal
      execution.
    """
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back

    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    
    cur_frame = inspect.stack()[1]
    banner = 'Interacting at File "{0}", line {1} \n'
    banner += '    Unix:    <Control>-D continues the program; \n'
    banner += '    Windows: <Control>-z <Enter> continues the program; \n'
    banner += '    exit() exits the program'
    
    code.interact(banner.format(cur_frame[1], cur_frame[2]), None, namespace)
