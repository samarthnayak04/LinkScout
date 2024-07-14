import asyncio
import vt

class APIError(Exception):
    pass

async def check_url(api_key, url):
    async with vt.Client(api_key) as client:
        try:
            url_id = vt.url_id(url)
            analysis = await client.get_object_async(f"/urls/{url_id}")

            stats = analysis.last_analysis_stats
            malicious = stats.get("malicious", 0)
            suspicious = stats.get("suspicious", 0)

            if malicious > 1 or suspicious > 1:
                result = f"The URL {url} is potentially malicious."
                result += f" Malicious detections: {malicious}, Suspicious detections: {suspicious}"
            else:
                result = f"The URL {url} appears to be safe."

            return result

        except vt.error.APIError as e:
            result = f"The URL {url} is Malicious."
            return result
            # raise APIError(f"An error occurred: {e}")
        except Exception as e:
            # raise APIError(f"An unexpected error occurred: {e}")
            result = f"The URL {url} is Malicious."
            return result
