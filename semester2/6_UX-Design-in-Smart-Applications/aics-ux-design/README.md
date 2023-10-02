# aics-ux-design

aics-ux-design

## FRONTEND

### Frontend Libraries used

- Angular: https://angular.io
- Angular Material: https://material.angular.io

### run 1st time

```
cd news-recommender-frontend
npm install
npm run start
```

### run any other time

```
cd news-recommender-frontend
npm run start
```

## BACKEND

```
cd news-recommender-backend
poetry install
poetry run uvicorn news_recommender_backend.main_server:main_server --reload --env-file=.env
```

### Backend Libraries used

- FastAPI: https://fastapi.tiangolo.com/
- SQLAlchemy: https://www.sqlalchemy.org/
- tensorflow: https://www.tensorflow.org/

### run 1st time

1. Install poetry build tool: https://python-poetry.org/docs/#installation

2. setup mysql database and run script:

   ```
   CREATE DATABASE IF NOT EXISTS newsrecommenderdb;
   USE newsrecommenderdb;
   CREATE USER IF NOT EXISTS 'newsrecommenderdbUser'@'%' identified by 'newsrecommenderdbPassword';
   GRANT ALL ON newsrecommenderdb.* to 'newsrecommenderdbUser'@'%';
   ```

3. run db migrations:

   ```
   poetry run alembic revision --autogenerate -m "some message"
   poetry run alembic upgrade head
   ```

4. run backend cli

   ```
   cd news-recommender-backend
   poetry install
   poetry run python news_recommender_backend/main_cli.py nn create --name default-nn --active
   # the following command needs added users with filled Questionnaire in the db
   poetry run python news_recommender_backend/main_cli.py nn train --name default-nn
   ```

5. run backend server

   ```
   poetry run uvicorn news_recommender_backend.main_server:main_server --reload --env-file=.env
   ```

### run any other time

```
cd news-recommender-backend
poetry run uvicorn news_recommender_backend.main_server:main_server --reload --env-file=.env
```
