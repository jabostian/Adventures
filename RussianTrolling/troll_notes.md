### Notes from Running the Russian Troll Tweet Analysis
Python notebook to explore the tweets:
- https://github.com/jswest/russian-troll-tweets-exploration

Tweets in csv form:
- https://github.com/fivethirtyeight/russian-troll-tweets/

#### Setup
Clone the repo for the code, and download the zip file of the text (csv)
files.  Put the tweets in a directory within the code repo.
```
git clone https://github.com/jswest/russian-troll-tweets-exploration.git
<download russian-troll-tweets-master.zip>
mv ~/Desktop/russian-troll-tweets-master.zip ~/git/russian-troll-tweets-exploration
```

Build an environment to run the model notebook
```
conda create -n troll python nodejs gensim
source activate troll
pip install -r requirements.txt
pip install --upgrade pip
npm install -g http-server
```

Run the notebook server and open the model-topics notebook from the
_**russian-troll-tweets-exploration**_ repo.
