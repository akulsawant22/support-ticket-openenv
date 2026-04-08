from fastapi import FastAPI
from envs.support_env.server.app import app as existing_app

app = existing_app
