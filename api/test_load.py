from locust import HttpUser, task, between

class ClaimAPIUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task(10)
    def predict_claim(self):
        self.client.post("/predict", json={
            "sentence": "The Empire State Building was completed in 1931."
        })

    @task(3)
    def predict_opinion(self):
        self.client.post("/predict", json={
            "sentence": "I think pizza is the best food."
        })

    @task(1)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def batch_predict(self):
        self.client.post("/predict/batch", json=[
            "Water boils at 100 degrees Celsius.",
            "Mount Everest is the tallest mountain.",
            "The moon is made of cheese.",
        ])