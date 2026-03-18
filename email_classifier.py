"""
AI Email Classifier - Hybrid ML + LLM Pipeline
Author: Joseph Davis
"""
import csv, json, logging, re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional
import joblib, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("email_classifier")
MODEL_DIR = Path("models"); MODEL_DIR.mkdir(exist_ok=True)

class Category(str, Enum):
    URGENT="urgent"; MEETING="meeting"; NEWSLETTER="newsletter"
    SUPPORT="support"; SALES="sales"; PERSONAL="personal"; SPAM="spam"; OTHER="other"

@dataclass
class Email:
    subject: str; body: str; sender: str = ""
    @property
    def full_text(self): return f"{self.subject} {self.body}"
    def clean(self):
        t = self.full_text.lower()
        t = re.sub(r"http\\S+", "[URL]", t)
        t = re.sub(r"\\S+@\\S+", "[EMAIL]", t)
        return re.sub(r"\\s+", " ", re.sub(r"[^\\w\\s\\[\\]]", " ", t)).strip()

@dataclass
class Result:
    category: Category; confidence: float; method: str
    scores: dict = field(default_factory=dict)
    @property
    def is_confident(self): return self.confidence >= 0.7
    def to_dict(self): return {"category":self.category.value,"confidence":round(self.confidence,4),"method":self.method}

class TrainingData:
    TEMPLATES = {
        Category.URGENT: [("URGENT: Server down","Production not responding. Immediate action required."),
            ("Critical: Deploy failed","Build pipeline failed on main. All services affected."),
            ("ACTION REQUIRED: Security breach","Suspicious login from unknown IP."),
            ("ASAP: Client escalation","Major client threatening to cancel."),
            ("Emergency: Data loss","Database replication causing inconsistencies.")],
        Category.MEETING: [("Team standup Monday","Let's sync on sprint progress."),
            ("1:1 rescheduled","Moving catch-up to Thursday 2pm."),
            ("Q4 Planning Workshop","Block 2 hours for quarterly planning."),
            ("Interview: Backend Dev","Candidate scheduled for 3pm."),
            ("Board meeting prep","Review attached deck before tomorrow.")],
        Category.NEWSLETTER: [("Weekly Python Digest","Top articles: FastAPI, async patterns."),
            ("TechCrunch Daily","AI funding roundup, startup news."),
            ("GitHub weekly report","23 contributions across 5 repos."),
            ("Product Hunt Digest","Top products: AI assistant, dev tools."),
            ("Medium Daily","Stories based on your reading history.")],
        Category.SUPPORT: [("Re: Ticket #4521","Still can't access after password reset."),
            ("Bug: Export broken","CSV export empty for 1000+ rows."),
            ("Feature request: Dark mode","Would love dark mode in dashboard."),
            ("API rate limiting","Getting 429 errors over 10 req/min."),
            ("Billing question","Charged twice for March subscription.")],
        Category.SALES: [("Partnership opportunity","Explore potential integration."),
            ("50% off annual plan","Upgrade now and save."),
            ("Demo follow up","When works for a 30-min demo?"),
            ("ROI case study","Company X increased productivity 40%."),
            ("Exclusive beta access","Selected for early access to AI features.")],
        Category.PERSONAL: [("Happy Birthday!","Let's celebrate this weekend."),
            ("Dinner Saturday?","Try that new Italian place?"),
            ("Trip photos","Best shots from hiking weekend."),
            ("Congrats on new job!","Well deserved. Let's catch up."),
            ("Family reunion","Looking at dates in August.")],
        Category.SPAM: [("Won $1,000,000!","Claim your prize now."),
            ("Free iPhone 15","Selected for exclusive giveaway."),
            ("Hot singles nearby","Meet people tonight."),
            ("Earn $5000/day","Simple trick, no experience needed."),
            ("Account compromised","Click to verify identity immediately.")],
    }
    @classmethod
    def generate(cls):
        data = []
        for cat, templates in cls.TEMPLATES.items():
            for subj, body in templates:
                data.append((Email(subject=subj, body=body).clean(), cat.value))
        return data

class MLClassifier:
    def __init__(self):
        self.pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")),
                                  ("clf", MultinomialNB(alpha=0.1))])
        self.trained = False
    def train(self, texts, labels):
        if len(texts) > 10:
            Xt, Xv, yt, yv = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
        else: Xt, yt, Xv, yv = texts, labels, texts, labels
        self.pipeline.fit(Xt, yt); self.trained = True
        logger.info(f"Trained on {len(Xt)} samples")
        logger.info(classification_report(yv, self.pipeline.predict(Xv), zero_division=0))
    def classify(self, email):
        proba = self.pipeline.predict_proba([email.clean()])[0]
        idx = np.argmax(proba)
        return Result(Category(self.pipeline.classes_[idx]), float(proba[idx]), "ml",
                     {c:float(s) for c,s in zip(self.pipeline.classes_, proba)})
    def save(self, path=None):
        joblib.dump(self.pipeline, path or MODEL_DIR/"classifier.joblib")
    def load(self, path=None):
        self.pipeline = joblib.load(path or MODEL_DIR/"classifier.joblib"); self.trained = True

def main():
    print("="*60 + "\
 AI Email Classifier Demo\
" + "="*60)
    data = TrainingData.generate()
    texts, labels = [d[0] for d in data], [d[1] for d in data]
    print(f"Training on {len(data)} samples...")
    clf = MLClassifier(); clf.train(texts, labels); clf.save()
    test = [Email("Server down!","Production API 500 errors since 3am."),
            Email("Lunch tomorrow?","Grab sushi at noon? My treat!"),
            Email("FREE GIFT!","Click to claim exclusive $500 reward."),
            Email("Sprint Review Thursday","Review completed stories."),
            Email("Can't login","Password reset not working.")]
    print("\
Classification Results:")
    for e in test:
        r = clf.classify(e)
        bar = chr(9608) * int(r.confidence * 20)
        print(f"  [{r.category.value:>12}] {bar} {r.confidence:.0%}  {e.subject}")

if __name__ == "__main__": main()
