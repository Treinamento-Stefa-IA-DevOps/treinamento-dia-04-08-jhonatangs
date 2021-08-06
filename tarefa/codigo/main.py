import pickle

from fastapi import FastAPI
from pydantic import BaseModel


class Person(BaseModel):
    Sex: int
    Age: float
    Lifeboat: int
    Pclass: int


app = FastAPI()


@app.post("/model")
## Coloque seu codigo na função abaixo
def titanic(person: Person):
    with open("model/Titanic.pkl", "rb") as fid:
        try:
            titanic = pickle.load(fid)
            y_pred = bool(
                titanic.predict(
                    [[person.Sex, person.Age, person.Lifeboat, person.Pclass]]
                )[0]
            )

            return {
                "survived": y_pred,
                "status": 200,
                "message": "Survived" if y_pred else "Did not survive",
            }

        except Exception as e:
            return {"survived": None, "status": 500, "message": e}


@app.get("/model")
def get():
    return {"hello": "test"}


@app.get("/")
def root():
    return {"message": "Hello, Titanic!"}
