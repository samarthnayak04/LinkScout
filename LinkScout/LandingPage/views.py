from django.shortcuts import render, redirect
from django.urls import reverse
from .REST_api import check_url, APIError
import asyncio

def process_url_input(request):
    if request.method == 'POST':
        url_input = request.POST.get('url_input')
        api_key = '0fe0f18692802e458feb4d2370d6ab165eaa37b21673bcc759b13b64fb0bc991'

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(check_url(api_key, url_input))
        except APIError as e:
            result = str(e)
        except Exception as e:
            result = "An unexpected error occurred."
            # Log the exception for debugging purposes
            print(f"Unexpected error in process_url_input: {e}")
        finally:
            loop.close()

        # Redirect to the result page with the result in the query parameters
        return redirect(f"{reverse('url_result')}?result={result}")

    return render(request, 'index.html')

def url_result(request):
    result = request.GET.get('result', 'No result found.')
    return render(request, 'result.html', {'result': result})
