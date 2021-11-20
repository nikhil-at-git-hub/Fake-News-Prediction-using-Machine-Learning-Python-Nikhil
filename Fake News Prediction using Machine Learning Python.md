
# Fake News Prediction using Machine Learning Python

## News prediction machine learning model that predicts whether a news is real or fake!

## 1: Fake news
## 2: Real news


```python
# Importing the dependencies
```


```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```


```python
# Stopwords are words that do not add much value to our data such as i, me, we, our, myself etc.
# PorterStemmer is used to extract out rootword from the main word(actress, actor,acting: act is the root word)
# TfidfVectorizer is used to convert textual data into numerical data that the system can understand
```


```python
import nltk
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     C:\Users\nvnik\AppData\Roaming\nltk_data...
    [nltk_data]   Unzipping corpora\stopwords.zip.
    




    True




```python
print(stopwords.words('english'))
```

    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    

## Data Preprocessing


```python
# Loding the dataset in pandas Dataframe
```


```python
dataset = pd.read_csv('C:/Users/nvnik/Desktop/Fake News Prediction-Python/train.csv')
```


```python
dataset.shape
```




    (20800, 5)




```python
dataset.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>author</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>Darrell Lucus</td>
      <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
      <td>Daniel J. Flynn</td>
      <td>Ever get the feeling your life circles the rou...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Why the Truth Might Get You Fired</td>
      <td>Consortiumnews.com</td>
      <td>Why the Truth Might Get You Fired October 29, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>15 Civilians Killed In Single US Airstrike Hav...</td>
      <td>Jessica Purkiss</td>
      <td>Videos 15 Civilians Killed In Single US Airstr...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Iranian woman jailed for fictional unpublished...</td>
      <td>Howard Portnoy</td>
      <td>Print \nAn Iranian woman has been sentenced to...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
dataset.isnull().sum()
```




    id           0
    title      558
    author    1957
    text        39
    label        0
    dtype: int64




```python
dataset.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>author</th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20800.000000</td>
      <td>20242</td>
      <td>18843</td>
      <td>20761</td>
      <td>20800.000000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>19803</td>
      <td>4201</td>
      <td>20386</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>Get Ready For Civil Unrest: Survey Finds That ...</td>
      <td>Pam Key</td>
      <td></td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>5</td>
      <td>243</td>
      <td>75</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10399.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.500625</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6004.587135</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.500012</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5199.750000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>10399.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>15599.250000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>20799.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.options.display.max_columns = None
```


```python
import seaborn as sns
import matplotlib.pyplot as plt
!matplotlib inline

plt.figure(figsize=(5,5))


sns.heatmap(dataset.isnull())
```

    'matplotlib' is not recognized as an internal or external command,
    operable program or batch file.
    




    <matplotlib.axes._subplots.AxesSubplot at 0x1f3501e1f60>




![png](output_16_2.png)



```python
dataset=dataset.fillna('')
```


```python
sns.heatmap(dataset.isnull())

```




    <matplotlib.axes._subplots.AxesSubplot at 0x1f351be05f8>




![png](output_18_1.png)



```python
dataset.isnull().sum()

```




    id        0
    title     0
    author    0
    text      0
    label     0
    dtype: int64




```python
# Merging the author name and news title column in new cloumn which is content
dataset['content'] = dataset['author']+' '+dataset['title']
```


```python
print(dataset['content'])
```

    0        Darrell Lucus House Dem Aide: We Didn’t Even S...
    1        Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...
    2        Consortiumnews.com Why the Truth Might Get You...
    3        Jessica Purkiss 15 Civilians Killed In Single ...
    4        Howard Portnoy Iranian woman jailed for fictio...
    5        Daniel Nussbaum Jackie Mason: Hollywood Would ...
    6         Life: Life Of Luxury: Elton John’s 6 Favorite...
    7        Alissa J. Rubin Benoît Hamon Wins French Socia...
    8         Excerpts From a Draft Script for Donald Trump...
    9        Megan Twohey and Scott Shane A Back-Channel Pl...
    10       Aaron Klein Obama’s Organizing for Action Part...
    11       Chris Tomlinson BBC Comedy Sketch "Real Housew...
    12       Amando Flavio Russian Researchers Discover Sec...
    13       Jason Ditz US Officials See No Link Between Tr...
    14       AnotherAnnie Re: Yes, There Are Paid Governmen...
    15       Jack Williams In Major League Soccer, Argentin...
    16       Michael Corkery and Stacy Cowley Wells Fargo C...
    17       Starkman Anonymous Donor Pays $2.5 Million To ...
    18                       The Doc FBI Closes In On Hillary!
    19       Jeff Poor Chuck Todd: ’BuzzFeed Did Donald Tru...
    20        News: Hope For The GOP: A Nude Paul Ryan Has ...
    21       Jerome Hudson Monica Lewinsky, Clinton Sex Sca...
    22       Pam Key Rob Reiner: Trump Is ’Mentally Unstabl...
    23        Massachusetts Cop’s Wife Busted for Pinning F...
    24       Donald G. McNeil Jr. and Pam Belluck Abortion ...
    25       Ira Helfand Nukes and the UN: a Historic Treat...
    26       Aaron Klein and Ali Waked EXCLUSIVE: Islamic S...
    27       Amanda Shea Humiliated Hillary Tries To Hide W...
    28       Jim Dwyer Andrea Tantaros of Fox News Claims R...
    29       Mark Landler How Hillary Clinton Became a Hawk...
                                   ...                        
    20770    Iron Sheik HUMA ABEDIN SWORE UNDER OATH SHE GA...
    20771                                          Letsbereal 
    20772                                         beersession 
    20773    Vicki Batts Australia to hunt down anti-vax nu...
    20774    Liam Deacon Government Report: Islamists Build...
    20775    Arnaldo Rodgers How this WWII airman is helpin...
    20776    Jameson Parker Trump Campaign Says Hillary Sup...
    20777    admin Editor of Austria’s Largest Paper Charge...
    20778    Neil Irwin This Is a Jobs Report That Democrat...
    20779    Thomas D. Williams, Ph.D. Christians in 2017 ’...
    20780    Christine Hauser Florida Woman Charged in Deat...
    20781    Heather Callaghan Time is Running Out to Stop ...
    20782    The Doc The Fix Is In: NBC Affiliate Accidenta...
    20783    Charles McDermid Samsung, Kim Jong-un, Rex Til...
    20784    Debbie Menon Comment on World Heaves Sigh of R...
    20785    Ann Coulter Ann Coulter: How to Provide Univer...
    20786     Government Forces Advancing at Damascus-Alepp...
    20787    Ian Mason Sally Yates Won’t Say If Trump Was W...
    20788    Joe Clark Maine’s Gov. LePage Threatens To ‘In...
    20789    Warner Todd Huston Sen. McConnell: The Supreme...
    20790    Adam Shaw Nikki Haley Blasts U.N. Human Rights...
    20791    Daniel Greenfield Lawyer Who Kept Hillary Camp...
    20792    John Hayward Jakarta Bombing Kills Three Polic...
    20793    Robert Rich Idiot Who Destroyed Trump Hollywoo...
    20794    Lee Stranahan Trump: Putin ’Very Smart’ to Not...
    20795    Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...
    20796    Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...
    20797    Michael J. de la Merced and Rachel Abrams Macy...
    20798    Alex Ansary NATO, Russia To Hold Parallel Exer...
    20799              David Swanson What Keeps the F-35 Alive
    Name: content, Length: 20800, dtype: object
    


```python
# # separting the data(content column) and label column
# axis=1 removes column axis=0 removes row

X = dataset.drop(columns='label', axis=1)
Y = dataset['label']
```


```python
print (X)
print (Y)
```

              id                                              title  \
    0          0  House Dem Aide: We Didn’t Even See Comey’s Let...   
    1          1  FLYNN: Hillary Clinton, Big Woman on Campus - ...   
    2          2                  Why the Truth Might Get You Fired   
    3          3  15 Civilians Killed In Single US Airstrike Hav...   
    4          4  Iranian woman jailed for fictional unpublished...   
    5          5  Jackie Mason: Hollywood Would Love Trump if He...   
    6          6  Life: Life Of Luxury: Elton John’s 6 Favorite ...   
    7          7  Benoît Hamon Wins French Socialist Party’s Pre...   
    8          8  Excerpts From a Draft Script for Donald Trump’...   
    9          9  A Back-Channel Plan for Ukraine and Russia, Co...   
    10        10  Obama’s Organizing for Action Partners with So...   
    11        11  BBC Comedy Sketch "Real Housewives of ISIS" Ca...   
    12        12  Russian Researchers Discover Secret Nazi Milit...   
    13        13  US Officials See No Link Between Trump and Russia   
    14        14  Re: Yes, There Are Paid Government Trolls On S...   
    15        15  In Major League Soccer, Argentines Find a Home...   
    16        16  Wells Fargo Chief Abruptly Steps Down - The Ne...   
    17        17  Anonymous Donor Pays $2.5 Million To Release E...   
    18        18                          FBI Closes In On Hillary!   
    19        19  Chuck Todd: ’BuzzFeed Did Donald Trump a Polit...   
    20        20  News: Hope For The GOP: A Nude Paul Ryan Has J...   
    21        21  Monica Lewinsky, Clinton Sex Scandal Set for ’...   
    22        22  Rob Reiner: Trump Is ’Mentally Unstable’ - Bre...   
    23        23  Massachusetts Cop’s Wife Busted for Pinning Fa...   
    24        24  Abortion Pill Orders Rise in 7 Latin American ...   
    25        25  Nukes and the UN: a Historic Treaty to Ban Nuc...   
    26        26  EXCLUSIVE: Islamic State Supporters Vow to ‘Sh...   
    27        27  Humiliated Hillary Tries To Hide What Camera C...   
    28        28  Andrea Tantaros of Fox News Claims Retaliation...   
    29        29  How Hillary Clinton Became a Hawk - The New Yo...   
    ...      ...                                                ...   
    20770  20770  HUMA ABEDIN SWORE UNDER OATH SHE GAVE UP ‘ALL ...   
    20771  20771                                                      
    20772  20772                                                      
    20773  20773  Australia to hunt down anti-vax nurses and pro...   
    20774  20774  Government Report: Islamists Building ’Paralle...   
    20775  20775  How this WWII airman is helping veterans heal ...   
    20776  20776  Trump Campaign Says Hillary Supporter Tried As...   
    20777  20777  Editor of Austria’s Largest Paper Charged with...   
    20778  20778  This Is a Jobs Report That Democrats Can Boast...   
    20779  20779  Christians in 2017 ’Most Persecuted Group in t...   
    20780  20780  Florida Woman Charged in Death of Infant in ‘C...   
    20781  20781  Time is Running Out to Stop Kratom Ban – Need ...   
    20782  20782  The Fix Is In: NBC Affiliate Accidentally Post...   
    20783  20783  Samsung, Kim Jong-un, Rex Tillerson: Your Morn...   
    20784  20784  Comment on World Heaves Sigh of Relief after T...   
    20785  20785  Ann Coulter: How to Provide Universal Health C...   
    20786  20786  Government Forces Advancing at Damascus-Aleppo...   
    20787  20787  Sally Yates Won’t Say If Trump Was Wiretapped ...   
    20788  20788  Maine’s Gov. LePage Threatens To ‘Investigate’...   
    20789  20789  Sen. McConnell: The Supreme Court Vacancy Was ...   
    20790  20790  Nikki Haley Blasts U.N. Human Rights Office fo...   
    20791  20791  Lawyer Who Kept Hillary Campaign Chief Out of ...   
    20792  20792  Jakarta Bombing Kills Three Police Officers, L...   
    20793  20793  Idiot Who Destroyed Trump Hollywood Star Gets ...   
    20794  20794  Trump: Putin ’Very Smart’ to Not Retaliate ove...   
    20795  20795  Rapper T.I.: Trump a ’Poster Child For White S...   
    20796  20796  N.F.L. Playoffs: Schedule, Matchups and Odds -...   
    20797  20797  Macy’s Is Said to Receive Takeover Approach by...   
    20798  20798  NATO, Russia To Hold Parallel Exercises In Bal...   
    20799  20799                          What Keeps the F-35 Alive   
    
                                              author  \
    0                                  Darrell Lucus   
    1                                Daniel J. Flynn   
    2                             Consortiumnews.com   
    3                                Jessica Purkiss   
    4                                 Howard Portnoy   
    5                                Daniel Nussbaum   
    6                                                  
    7                                Alissa J. Rubin   
    8                                                  
    9                   Megan Twohey and Scott Shane   
    10                                   Aaron Klein   
    11                               Chris Tomlinson   
    12                                 Amando Flavio   
    13                                    Jason Ditz   
    14                                  AnotherAnnie   
    15                                 Jack Williams   
    16              Michael Corkery and Stacy Cowley   
    17                                      Starkman   
    18                                       The Doc   
    19                                     Jeff Poor   
    20                                                 
    21                                 Jerome Hudson   
    22                                       Pam Key   
    23                                                 
    24          Donald G. McNeil Jr. and Pam Belluck   
    25                                   Ira Helfand   
    26                     Aaron Klein and Ali Waked   
    27                                   Amanda Shea   
    28                                     Jim Dwyer   
    29                                  Mark Landler   
    ...                                          ...   
    20770                                 Iron Sheik   
    20771                                 Letsbereal   
    20772                                beersession   
    20773                                Vicki Batts   
    20774                                Liam Deacon   
    20775                            Arnaldo Rodgers   
    20776                             Jameson Parker   
    20777                                      admin   
    20778                                 Neil Irwin   
    20779                  Thomas D. Williams, Ph.D.   
    20780                           Christine Hauser   
    20781                          Heather Callaghan   
    20782                                    The Doc   
    20783                           Charles McDermid   
    20784                               Debbie Menon   
    20785                                Ann Coulter   
    20786                                              
    20787                                  Ian Mason   
    20788                                  Joe Clark   
    20789                         Warner Todd Huston   
    20790                                  Adam Shaw   
    20791                          Daniel Greenfield   
    20792                               John Hayward   
    20793                                Robert Rich   
    20794                              Lee Stranahan   
    20795                              Jerome Hudson   
    20796                           Benjamin Hoffman   
    20797  Michael J. de la Merced and Rachel Abrams   
    20798                                Alex Ansary   
    20799                              David Swanson   
    
                                                        text  \
    0      House Dem Aide: We Didn’t Even See Comey’s Let...   
    1      Ever get the feeling your life circles the rou...   
    2      Why the Truth Might Get You Fired October 29, ...   
    3      Videos 15 Civilians Killed In Single US Airstr...   
    4      Print \nAn Iranian woman has been sentenced to...   
    5      In these trying times, Jackie Mason is the Voi...   
    6      Ever wonder how Britain’s most iconic pop pian...   
    7      PARIS  —   France chose an idealistic, traditi...   
    8      Donald J. Trump is scheduled to make a highly ...   
    9      A week before Michael T. Flynn resigned as nat...   
    10     Organizing for Action, the activist group that...   
    11     The BBC produced spoof on the “Real Housewives...   
    12     The mystery surrounding The Third Reich and Na...   
    13     Clinton Campaign Demands FBI Affirm Trump's Ru...   
    14     Yes, There Are Paid Government Trolls On Socia...   
    15     Guillermo Barros Schelotto was not the first A...   
    16     The scandal engulfing Wells Fargo toppled its ...   
    17     A Caddo Nation tribal leader has just been fre...   
    18     FBI Closes In On Hillary! Posted on Home » Hea...   
    19     Wednesday after   Donald Trump’s press confere...   
    20     Email \nSince Donald Trump entered the electio...   
    21     Screenwriter Ryan Murphy, who has produced the...   
    22     Sunday on MSNBC’s “AM Joy,” actor and director...   
    23     Massachusetts Cop’s Wife Busted for Pinning Fa...   
    24     Orders for abortion pills by women in seven La...   
    25     Email \nIn an historic move the United Nations...   
    26     JERUSALEM  —   Islamic State sympathizers and ...   
    27     Humiliated Hillary Tries To Hide What Camera C...   
    28     Andrea Tantaros, a former Fox News host, charg...   
    29     Hillary Clinton sat in the hideaway study off ...   
    ...                                                  ...   
    20770  Home › POLITICS | US NEWS › HUMA ABEDIN SWORE ...   
    20771  DYN's Statement on Last Week's Botnet Attack h...   
    20772  Kinda reminds me of when Carter gave away the ...   
    20773  Australia to hunt down anti-vax nurses and pro...   
    20774  Aided by a politically correct culture of “tol...   
    20775  ‹ › Arnaldo Rodgers is a trained and educated ...   
    20776  Donald Trump was rushed from a rally stage by ...   
    20777  Breitbart October 26, 2016 \nAn editor of Aust...   
    20778  There’s not much to say about the July jobs nu...   
    20779  In many parts of the world, Christians gatheri...   
    20780  Early on Oct. 6, Erin   was awakened by the so...   
    20781  By Brandon Turbeville When the DEA announced t...   
    20782  Home » Headlines » World News » The Fix Is In:...   
    20783  Good morning.  Here’s what you need to know: •...   
    20784    Finian Cunningham has written extensively on...   
    20785  The first sentence of Congress’ Obamacare repe...   
    20786  #FROMTHEFRONT #MAPS 22.11.2016 - 1,361 views 5...   
    20787  Former Deputy Attorney General Sally Yates dec...   
    20788  Google Pinterest Digg Linkedin Reddit Stumbleu...   
    20789  Senate Majority Leader Mitch McConnell (R, KY)...   
    20790  U. S Ambassador to the United Nations Nikki Ha...   
    20791  Lawyer Who Kept Hillary Campaign Chief Out of ...   
    20792  Two suicide bombers attacked a bus station in ...   
    20793  Share This \nAlthough the vandal who thought i...   
    20794  Donald Trump took to Twitter Friday to praise ...   
    20795  Rapper T. I. unloaded on black celebrities who...   
    20796  When the Green Bay Packers lost to the Washing...   
    20797  The Macy’s of today grew from the union of sev...   
    20798  NATO, Russia To Hold Parallel Exercises In Bal...   
    20799    David Swanson is an author, activist, journa...   
    
                                                     content  
    0      Darrell Lucus House Dem Aide: We Didn’t Even S...  
    1      Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...  
    2      Consortiumnews.com Why the Truth Might Get You...  
    3      Jessica Purkiss 15 Civilians Killed In Single ...  
    4      Howard Portnoy Iranian woman jailed for fictio...  
    5      Daniel Nussbaum Jackie Mason: Hollywood Would ...  
    6       Life: Life Of Luxury: Elton John’s 6 Favorite...  
    7      Alissa J. Rubin Benoît Hamon Wins French Socia...  
    8       Excerpts From a Draft Script for Donald Trump...  
    9      Megan Twohey and Scott Shane A Back-Channel Pl...  
    10     Aaron Klein Obama’s Organizing for Action Part...  
    11     Chris Tomlinson BBC Comedy Sketch "Real Housew...  
    12     Amando Flavio Russian Researchers Discover Sec...  
    13     Jason Ditz US Officials See No Link Between Tr...  
    14     AnotherAnnie Re: Yes, There Are Paid Governmen...  
    15     Jack Williams In Major League Soccer, Argentin...  
    16     Michael Corkery and Stacy Cowley Wells Fargo C...  
    17     Starkman Anonymous Donor Pays $2.5 Million To ...  
    18                     The Doc FBI Closes In On Hillary!  
    19     Jeff Poor Chuck Todd: ’BuzzFeed Did Donald Tru...  
    20      News: Hope For The GOP: A Nude Paul Ryan Has ...  
    21     Jerome Hudson Monica Lewinsky, Clinton Sex Sca...  
    22     Pam Key Rob Reiner: Trump Is ’Mentally Unstabl...  
    23      Massachusetts Cop’s Wife Busted for Pinning F...  
    24     Donald G. McNeil Jr. and Pam Belluck Abortion ...  
    25     Ira Helfand Nukes and the UN: a Historic Treat...  
    26     Aaron Klein and Ali Waked EXCLUSIVE: Islamic S...  
    27     Amanda Shea Humiliated Hillary Tries To Hide W...  
    28     Jim Dwyer Andrea Tantaros of Fox News Claims R...  
    29     Mark Landler How Hillary Clinton Became a Hawk...  
    ...                                                  ...  
    20770  Iron Sheik HUMA ABEDIN SWORE UNDER OATH SHE GA...  
    20771                                        Letsbereal   
    20772                                       beersession   
    20773  Vicki Batts Australia to hunt down anti-vax nu...  
    20774  Liam Deacon Government Report: Islamists Build...  
    20775  Arnaldo Rodgers How this WWII airman is helpin...  
    20776  Jameson Parker Trump Campaign Says Hillary Sup...  
    20777  admin Editor of Austria’s Largest Paper Charge...  
    20778  Neil Irwin This Is a Jobs Report That Democrat...  
    20779  Thomas D. Williams, Ph.D. Christians in 2017 ’...  
    20780  Christine Hauser Florida Woman Charged in Deat...  
    20781  Heather Callaghan Time is Running Out to Stop ...  
    20782  The Doc The Fix Is In: NBC Affiliate Accidenta...  
    20783  Charles McDermid Samsung, Kim Jong-un, Rex Til...  
    20784  Debbie Menon Comment on World Heaves Sigh of R...  
    20785  Ann Coulter Ann Coulter: How to Provide Univer...  
    20786   Government Forces Advancing at Damascus-Alepp...  
    20787  Ian Mason Sally Yates Won’t Say If Trump Was W...  
    20788  Joe Clark Maine’s Gov. LePage Threatens To ‘In...  
    20789  Warner Todd Huston Sen. McConnell: The Supreme...  
    20790  Adam Shaw Nikki Haley Blasts U.N. Human Rights...  
    20791  Daniel Greenfield Lawyer Who Kept Hillary Camp...  
    20792  John Hayward Jakarta Bombing Kills Three Polic...  
    20793  Robert Rich Idiot Who Destroyed Trump Hollywoo...  
    20794  Lee Stranahan Trump: Putin ’Very Smart’ to Not...  
    20795  Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...  
    20796  Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...  
    20797  Michael J. de la Merced and Rachel Abrams Macy...  
    20798  Alex Ansary NATO, Russia To Hold Parallel Exer...  
    20799            David Swanson What Keeps the F-35 Alive  
    
    [20800 rows x 5 columns]
    0        1
    1        0
    2        1
    3        1
    4        1
    5        0
    6        1
    7        0
    8        0
    9        0
    10       0
    11       0
    12       1
    13       1
    14       1
    15       0
    16       0
    17       1
    18       1
    19       0
    20       1
    21       0
    22       0
    23       1
    24       0
    25       1
    26       0
    27       1
    28       0
    29       0
            ..
    20770    1
    20771    1
    20772    1
    20773    1
    20774    0
    20775    1
    20776    1
    20777    1
    20778    0
    20779    0
    20780    0
    20781    1
    20782    1
    20783    0
    20784    1
    20785    0
    20786    1
    20787    0
    20788    1
    20789    0
    20790    0
    20791    1
    20792    0
    20793    1
    20794    0
    20795    0
    20796    0
    20797    0
    20798    1
    20799    1
    Name: label, Length: 20800, dtype: int64
    

## Stemming


```python
# Reducing the word to root word
```


```python
port_stem = PorterStemmer()
```


```python
# defining stemming function
def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


```


```python
dataset['content'] = dataset['content'].apply(stemming)
```


```python
print(dataset['content'])
```

    0        darrel lucu hous dem aid even see comey letter...
    1        daniel j flynn flynn hillari clinton big woman...
    2                   consortiumnew com truth might get fire
    3        jessica purkiss civilian kill singl us airstri...
    4        howard portnoy iranian woman jail fiction unpu...
    5        daniel nussbaum jacki mason hollywood would lo...
    6        life life luxuri elton john favorit shark pict...
    7        alissa j rubin beno hamon win french socialist...
    8        excerpt draft script donald trump q ampa black...
    9        megan twohey scott shane back channel plan ukr...
    10       aaron klein obama organ action partner soro li...
    11       chri tomlinson bbc comedi sketch real housew i...
    12       amando flavio russian research discov secret n...
    13              jason ditz us offici see link trump russia
    14       anotheranni ye paid govern troll social media ...
    15       jack william major leagu soccer argentin find ...
    16       michael corkeri staci cowley well fargo chief ...
    17       starkman anonym donor pay million releas every...
    18                                   doc fbi close hillari
    19       jeff poor chuck todd buzzfe donald trump polit...
    20       news hope gop nude paul ryan emerg ayahuasca t...
    21       jerom hudson monica lewinski clinton sex scand...
    22       pam key rob reiner trump mental unstabl breitbart
    23       massachusett cop wife bust pin fake home invas...
    24       donald g mcneil jr pam belluck abort pill orde...
    25       ira helfand nuke un histor treati ban nuclear ...
    26       aaron klein ali wake exclus islam state suppor...
    27       amanda shea humili hillari tri hide camera cau...
    28       jim dwyer andrea tantaro fox news claim retali...
    29       mark landler hillari clinton becam hawk new yo...
                                   ...                        
    20770    iron sheik huma abedin swore oath gave devic s...
    20771                                              letsber
    20772                                             beersess
    20773    vicki batt australia hunt anti vax nurs prosec...
    20774    liam deacon govern report islamist build paral...
    20775    arnaldo rodger wwii airman help veteran heal h...
    20776    jameson parker trump campaign say hillari supp...
    20777    admin editor austria largest paper charg hate ...
    20778    neil irwin job report democrat boast new york ...
    20779      thoma william ph christian persecut group world
    20780    christin hauser florida woman charg death infa...
    20781    heather callaghan time run stop kratom ban nee...
    20782    doc fix nbc affili accident post elect result ...
    20783    charl mcdermid samsung kim jong un rex tillers...
    20784    debbi menon comment world heav sigh relief tru...
    20785    ann coulter ann coulter provid univers health ...
    20786    govern forc advanc damascu aleppo highway east...
    20787     ian mason salli yate say trump wiretap breitbart
    20788    joe clark main gov lepag threaten investig col...
    20789    warner todd huston sen mcconnel suprem court v...
    20790    adam shaw nikki haley blast u n human right of...
    20791    daniel greenfield lawyer kept hillari campaign...
    20792    john hayward jakarta bomb kill three polic off...
    20793    robert rich idiot destroy trump hollywood star...
    20794    lee stranahan trump putin smart retali obama s...
    20795    jerom hudson rapper trump poster child white s...
    20796    benjamin hoffman n f l playoff schedul matchup...
    20797    michael j de la merc rachel abram maci said re...
    20798    alex ansari nato russia hold parallel exercis ...
    20799                            david swanson keep f aliv
    Name: content, Length: 20800, dtype: object
    


```python
# separting the data and label column

X = dataset['content'].values
Y = dataset['label'].values
```


```python
print(X)
```

    ['darrel lucu hous dem aid even see comey letter jason chaffetz tweet'
     'daniel j flynn flynn hillari clinton big woman campu breitbart'
     'consortiumnew com truth might get fire' ...
     'michael j de la merc rachel abram maci said receiv takeov approach hudson bay new york time'
     'alex ansari nato russia hold parallel exercis balkan'
     'david swanson keep f aliv']
    


```python
print(Y)
```

    [1 0 1 ... 0 1 1]
    


```python
# converting textual data to numerical data
# Tfidf: Term Frequency Inverse Document Frequency: number of times a word is repeated in data

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
```


```python
print(X)
```

      (0, 15686)	0.28485063562728646
      (0, 13473)	0.2565896679337957
      (0, 8909)	0.3635963806326075
      (0, 8630)	0.29212514087043684
      (0, 7692)	0.24785219520671603
      (0, 7005)	0.21874169089359144
      (0, 4973)	0.233316966909351
      (0, 3792)	0.2705332480845492
      (0, 3600)	0.3598939188262559
      (0, 2959)	0.2468450128533713
      (0, 2483)	0.3676519686797209
      (0, 267)	0.27010124977708766
      (1, 16799)	0.30071745655510157
      (1, 6816)	0.1904660198296849
      (1, 5503)	0.7143299355715573
      (1, 3568)	0.26373768806048464
      (1, 2813)	0.19094574062359204
      (1, 2223)	0.3827320386859759
      (1, 1894)	0.15521974226349364
      (1, 1497)	0.2939891562094648
      (2, 15611)	0.41544962664721613
      (2, 9620)	0.49351492943649944
      (2, 5968)	0.3474613386728292
      (2, 5389)	0.3866530551182615
      (2, 3103)	0.46097489583229645
      :	:
      (20797, 13122)	0.2482526352197606
      (20797, 12344)	0.27263457663336677
      (20797, 12138)	0.24778257724396507
      (20797, 10306)	0.08038079000566466
      (20797, 9588)	0.174553480255222
      (20797, 9518)	0.2954204003420313
      (20797, 8988)	0.36160868928090795
      (20797, 8364)	0.22322585870464118
      (20797, 7042)	0.21799048897828688
      (20797, 3643)	0.21155500613623743
      (20797, 1287)	0.33538056804139865
      (20797, 699)	0.30685846079762347
      (20797, 43)	0.29710241860700626
      (20798, 13046)	0.22363267488270608
      (20798, 11052)	0.4460515589182236
      (20798, 10177)	0.3192496370187028
      (20798, 6889)	0.32496285694299426
      (20798, 5032)	0.4083701450239529
      (20798, 1125)	0.4460515589182236
      (20798, 588)	0.3112141524638974
      (20798, 350)	0.28446937819072576
      (20799, 14852)	0.5677577267055112
      (20799, 8036)	0.45983893273780013
      (20799, 3623)	0.37927626273066584
      (20799, 377)	0.5677577267055112
    

## Splitting dataset to training and test data


```python
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.2, stratify=Y, random_state=2)
```

## Training the model: Logistic Regression


```python
model = LogisticRegression()
```


```python
model.fit(X_train,Y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



## Evaluation of the model


```python
# Accuracy Score on the training data

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)

```


```python
print('Accuracy Score of the trainig Data: ', training_data_accuracy*100)
```

    Accuracy Score of the trainig Data:  98.65985576923076
    


```python
# Accuracy Score on the test data

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)

```


```python
print('Accuracy Score of the test Data: ', test_data_accuracy*100)
```

    Accuracy Score of the test Data:  97.90865384615385
    

## Making a user interactive predictive system


```python
X_new = X_test[3]

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
    print("The news is real")
else:
    print("The news is fake")
```

    [0]
    The news is real
    


```python
print(Y_test[3])
```

    0
    

### The predicted value from the prediction model and the value of the test dataset matches, hence we can say that the model is performing really well.
