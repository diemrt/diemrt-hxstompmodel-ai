from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import sys
import os
from pathlib import Path

# Add parent directory to Python path to import hx_stomp_qa
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from hx_stomp_qa import HXStompQA

class ChatAPI(APIView):
    qa_system = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if ChatAPI.qa_system is None:
            ChatAPI.qa_system = HXStompQA()

    def post(self, request):
        try:
            question = request.data.get('question')
            if not question:
                return Response(
                    {'error': 'Question is required'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )

            response = self.qa_system.answer_question(question)
            return Response(response)
        except Exception as e:
            return Response(
                {'error': str(e)}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
