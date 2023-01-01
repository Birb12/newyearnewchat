import os
from twilio.rest import Client
from twilio.twiml.messaging_response import MessagingResponse
from flask import Flask, request, redirect


account_sid = ""
auth_token = ""

client = Client(account_sid, auth_token)

def run_notify(resolutions, phonenum):
    body = "Hi! This is NewYearNewChat to remind you about your resolutions: "

    for i in resolutions:
        body += "-" + i + " "
    message = client.messages.create(body=body, messaging_service_sid='MGf681960bb091aa4c9262ed305d69b82d', to=phonenum)
    print(message.sid)

