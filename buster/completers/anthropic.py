import logging
import os
from typing import Iterator, Optional

import instructor
import anthropic

from buster.completers import Completer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AnthropicCompleter(Completer):
    def __init__(self, completion_kwargs: dict, client_kwargs: Optional[dict] = None):
        """Initialize the AnthropicCompleter with completion and client keyword arguments.

        Args:
          completion_kwargs: A dictionary of keyword arguments to be used for completions.
          client_kwargs: An optional dictionary of keyword arguments to be used for the client.
        """
        # use default client if none passed
        self.completion_kwargs = completion_kwargs

        if client_kwargs is None:
            client_kwargs = {}

        self.async_client = instructor.from_anthropic(anthropic.AsyncAnthropic(**client_kwargs))
        self.sync_client = instructor.from_anthropic(anthropic.Anthropic(**client_kwargs))

    async def async_complete(self, system_prompt: str, user_input: str, completion_kwargs=None):
        """Given a prompt and user input, returns the generated message and error flag.

        Args:
          prompt: The prompt containing the formatted documents and instructions.
          user_input: The user input to be responded to.
          completion_kwargs: An optional dictionary of keyword arguments to override the default completion kwargs.

        Returns:
          A tuple containing the completed message and a boolean indicating if an error occurred.

        Raises:
          anthropic.BadRequestError: If the completion request is invalid.
          anthropic.RateLimitError: If the servers are overloaded.
        """
        # Uses default configuration if not overridden

        if completion_kwargs is None:
            completion_kwargs = self.completion_kwargs

        messages = [
            {"role": "user", 
             "content": user_input},
        ]

        try:
            error = False
            response = await self.async_client.chat.completions.create(system=system_prompt, messages=messages, max_tokens=4000, **completion_kwargs)
        except anthropic.BadRequestError:
            error = True
            logger.exception("BadRequestError to Anthropic API. See traceback:")
            error_message = "BadRequestError to Anthropic API"
            return error_message, error

        except anthropic.InternalServerError:
            error = True
            logger.exception("InternalServerError with the Anthropic API. See traceback:")
            error_message = "InternalServerError with the Anthropic API. See traceback:"
            return error_message, error

        except anthropic.RateLimitError:
            error = True
            logger.exception("RateLimitError from Anthropic. See traceback:")
            error_message = "RateLimitError, try again later"
            return error_message, error

        except Exception as e:
            error = True
            logger.exception("Some kind of error happened trying to generate the response. See traceback:")
            error_message = "Something went wrong with connecting with LLM provider"
            return error_message, error

        if completion_kwargs.get("stream") is True:
            # We are entering streaming mode, so here we're just wrapping the streamed
            # response to be easier to handle later
            async def answer_generator():
                async for chunk in response:
                    if chunk.type == "content_block_delta":
                        token = chunk.delta.text
                        token = "" if token is None else token
                        yield token
                    else:
                        continue

            return answer_generator(), error

        else:
            if completion_kwargs.get("response_model") is None:
                full_response: str = response.content[0].text
            else:
                logger.info(f"{response.model_dump_json(indent=2)=}")
                full_response = response
            return full_response, error

    def sync_complete(self, system_prompt: str, user_input: str, completion_kwargs=None) -> (str | Iterator, bool):
        """Given a prompt and user input, returns the generated message and error flag.

        Args:
          prompt: The prompt containing the formatted documents and instructions.
          user_input: The user input to be responded to.
          completion_kwargs: An optional dictionary of keyword arguments to override the default completion kwargs.

        Returns:
          A tuple containing the completed message and a boolean indicating if an error occurred.

        Raises:
          openai.BadRequestError: If the completion request is invalid.
          openai.RateLimitError: If the OpenAI servers are overloaded.
        """
        # Uses default configuration if not overridden

        if completion_kwargs is None:
            completion_kwargs = self.completion_kwargs

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            error = False
            response = self.sync_client.chat.completions.create(messages=messages, **completion_kwargs)
        except anthropic.BadRequestError:
            error = True
            logger.exception("Invalid request to OpenAI API. See traceback:")
            error_message = "Something went wrong while connecting with OpenAI, try again soon!"
            return error_message, error

        except anthropic.RateLimitError:
            error = True
            logger.exception("RateLimit error from OpenAI. See traceback:")
            error_message = "OpenAI servers seem to be overloaded, try again later!"
            return error_message, error

        except Exception as e:
            error = True
            logger.exception("Some kind of error happened trying to generate the response. See traceback:")
            error_message = "Something went wrong with connecting with OpenAI, try again soon!"
            return error_message, error

        if completion_kwargs.get("stream") is True:
            # We are entering streaming mode, so here we're just wrapping the streamed
            # openai response to be easier to handle later
            def answer_generator():
                for chunk in response:
                    token = chunk.choices[0].delta.content

                    # Always stream a string, openAI returns None on last token
                    token = "" if token is None else token

                    yield token

            return answer_generator(), error

        else:
            full_response: str = response.choices[0].message.content
            return full_response, error
