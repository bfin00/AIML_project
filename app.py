import sqlite3 as sql
import numpy as np

from flask import Flask, render_template, abort, request
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
app = Flask(__name__)
db_path = './db.sqlite'


def encode(text: str):
    text = text.split('.')
    text = list(map(lambda it: model.encode(it), text))
    text = np.stack(text, axis=0)
    text = text.mean(axis=0)

    return text




def get_similar_texts(embedding):
    with sql.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(f'SELECT id, title, embedding FROM articles')

        texts = cursor.fetchall()

        embeddings = []
        embeddings.extend([embedding])
        embeddings.extend([np.frombuffer(buffer[2], dtype=np.float32) for buffer in texts])

        similarities = cosine_similarity(embeddings)[0].argsort()[::-1][:6]

        links = [(title, id) for id, title, _ in texts if id in similarities]

        links.reverse()

        return links


def add_text_to_db(title, text, embedding):
    if title is None or len(title) == 0:
        title = 'Untitled'

    embedding = embedding.tobytes()

    with sql.connect(db_path) as conn:
        cursor = conn.cursor()

        query = 'INSERT INTO articles (title, text, embedding) values (?, ?, ?)'
        data = (title, text, embedding)

        cursor.execute(query, data)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        title = request.form['title']
        add_to_db = request.form.get('add') == 'on'

        embedding = encode(text)

        links = get_similar_texts(embedding)

        if add_to_db:
            add_text_to_db(title, text, embedding)

        return render_template(
            'index.html',
            title=title,
            text=text,
            links=links
        )

    return render_template('index.html')


@app.route('/articles/<int:id>')
def article(id):
    with sql.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute(f'SELECT title, text, embedding FROM articles WHERE id={id}')

        data = cursor.fetchall()

        if len(data) == 0:
            abort(404)

        title = data[0][0]
        text = data[0][1]
        embedding = np.frombuffer(data[0][2])

        return render_template('article.html', title=title, text=text, embedding=embedding)


@app.before_first_request
def update_db():
    with sql.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT id, text, embedding FROM articles")

        embeddings = []

        for row in tqdm(cursor, desc='Fetching data'):
            id = row[0]
            text = row[1]
            embedding = encode(text)
            embeddings.append((id, embedding))

        for embedding in tqdm(embeddings, desc='Applying embeddings'):
            query = "UPDATE articles SET embedding=? WHERE id=?"
            data = (embedding[1], embedding[0])
            cursor.execute(query, data)

        conn.commit()


if __name__ == '__main__':
    app.run(debug=True)
