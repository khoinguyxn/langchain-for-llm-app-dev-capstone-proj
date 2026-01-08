"""Voice Research Agent - LiveKit integration for voice-based research queries."""

from livekit import agents, rtc
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, room_io
from livekit.plugins import langchain, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from research_agent import research_agent

server = AgentServer()


class VoiceResearchAgent(Agent):
    """Agent with instructions for voice-based research assistance."""

    def __init__(self):
        super().__init__(
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You always cite your sources when providing information and provides answers to a research question logically.""",
        )


@server.rtc_session()
async def voice_research_agent_session(ctx: JobContext):
    """LiveKit RTC session handler for voice research agent."""
    session = AgentSession(
        stt="cartesia/ink-whisper",
        llm=langchain.LLMAdapter(graph=research_agent),
        tts="cartesia/sonic-3",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=VoiceResearchAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind
                    == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            )
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )

if __name__ == "__main__":
    agents.cli.run_app(server)