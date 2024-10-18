from zenml.client import Client

artifact = Client().get_artifact_version("06142289-d099-443d-8fca-76b300596ae3")
loaded_artifact = artifact.load()

print(loaded_artifact)
