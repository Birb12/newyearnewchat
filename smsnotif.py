import os
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request, redirect


account_sid = "AC37a9ba2f792d06bbd362c2b9fff90bd3"
auth_token = "90439c5188dc2e0ab4fb42b2a2d673a6"

client = Client(account_sid, auth_token)

def run_notify(resolutions, phonenum):
    body = "Hi! This is NewYearNewChat to remind you about your resolutions: "

    for i in resolutions:
        body += "-" + i + " "
    message = client.messages.create(body=body, messaging_service_sid='MGf681960bb091aa4c9262ed305d69b82d', to=phonenum)
    print(message.sid)

