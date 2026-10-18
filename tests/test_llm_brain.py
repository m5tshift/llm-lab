"""Unit tests for LocalLLMBrain prompt construction and multimodal signal integration."""

import unittest
from unittest.mock import patch, MagicMock
from src.llm_brain import LocalLLMBrain
from src.core.schemas import DetectedObject


class TestLocalLLMBrain(unittest.TestCase):
    def setUp(self):
        with patch.object(LocalLLMBrain, "_check_health", return_value=True):
            self.brain = LocalLLMBrain()

    @patch("requests.post")
    def test_standalone_request_contains_only_user_text(self, mock_post):
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"response": "Ответ"}
        self.brain.generate_reply(
            streamer_speech="Что изображено?",
            chat_mood="",
            visual_context="Красный квадрат",
            bot_personality="Отвечай кратко",
            use_stream_prompt=False,
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"]["prompt"],
            "Отвечай кратко\nЧто изображено?\nКрасный квадрат",
        )

    @patch("requests.post")
    def test_empty_standalone_request_does_not_invent_a_prompt(self, mock_post):
        with self.assertRaisesRegex(ValueError, "Введите запрос"):
            self.brain.generate_reply("", "", "", bot_personality="", use_stream_prompt=False)
        mock_post.assert_not_called()

    @patch("requests.post")
    def test_answer_stream_question_includes_all_multimodal_data(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "В чате зрители обсуждают тактику стримера, danilkagrahov передает привет, а на экране идет бой.",
            "prompt_eval_count": 120,
            "eval_count": 25,
            "eval_duration": 500000000,
        }
        mock_post.return_value = mock_response

        detected = [
            DetectedObject(class_name="person", confidence=0.92, box=[0, 0, 10, 10]),
            DetectedObject(class_name="tv", confidence=0.88, box=[10, 10, 20, 20]),
        ]
        speech_hist = ["Всем привет!", "Сейчас идем на мид", "Осторожно, сзади!"]
        chat_msgs = [
            "[09:56:00] danilkagrahov: Привет",
            "[09:56:14] VELIKOLEPNJIMAESTRO: А понимать не нужно просто наной димбычу LUL",
        ]
        chat_stats = {
            "positive_ratio": 0.2,
            "negative_ratio": 0.8,
            "neutral_ratio": 0.0,
            "total_messages": 2,
        }

        res = self.brain.answer_stream_question(
            user_question="Какие сообщения есть в чате?",
            streamer_speech="Осторожно, сзади!",
            chat_mood="НЕГАТИВНОЕ",
            visual_context="Игровой экран",
            channel_name="streamer_test",
            chat_history=[{"role": "user", "content": "Привет"}],
            recent_chat_messages=chat_msgs,
            scene_description="Игрок сидит перед монитором и играет в шутер",
            detected_objects=detected,
            speech_history=speech_hist,
            speech_language="russian",
            chat_stats=chat_stats,
            session_duration=45,
            stream_fps=14.5,
            vlm_description="На табло 2:1",
        )

        self.assertIn("danilkagrahov", res.reply)
        self.assertEqual(res.prompt_tokens, 120)
        self.assertEqual(res.completion_tokens, 25)

        # Inspect the exact prompt payload sent to Ollama
        call_args = mock_post.call_args
        sent_payload = call_args[1]["json"]
        sent_prompt = sent_payload["prompt"]

        # 1. Vision: scene description & YOLO objects
        self.assertIn("Игрок сидит перед монитором", sent_prompt)
        self.assertIn("person (92%)", sent_prompt)
        self.assertIn("tv (88%)", sent_prompt)
        self.assertIn("визуальный анализ): На табло 2:1", sent_prompt)

        # 2. Audio: latest speech & speech history
        self.assertIn("Осторожно, сзади!", sent_prompt)
        self.assertIn("Всем привет!", sent_prompt)
        self.assertIn("Сейчас идем на мид", sent_prompt)

        # 3. Chat: RuBERT mood, stats percentages, and raw IRC messages
        self.assertIn("НЕГАТИВНОЕ", sent_prompt)
        self.assertIn("Позитив: 20%", sent_prompt)
        self.assertIn("Негатив: 80%", sent_prompt)
        self.assertIn("[09:56:00] danilkagrahov: Привет", sent_prompt)
        self.assertIn(
            "[09:56:14] VELIKOLEPNJIMAESTRO: А понимать не нужно просто наной димбычу LUL",
            sent_prompt,
        )

        # 4. Telemetry: channel, session time, FPS
        self.assertIn("#streamer_test", sent_prompt)
        self.assertIn("45 сек", sent_prompt)
        self.assertIn("14.5 FPS", sent_prompt)

        # 5. User question
        self.assertIn("Какие сообщения есть в чате?", sent_prompt)

    @patch("requests.post")
    def test_generate_reply_includes_multimodal_context(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": "Красиво зашел, тащи раунд!",
            "prompt_eval_count": 80,
            "eval_count": 10,
            "eval_duration": 200000000,
        }
        mock_post.return_value = mock_response

        res = self.brain.generate_reply(
            streamer_speech="Заходим на точку!",
            chat_mood="ПОЗИТИВНОЕ",
            visual_context="",
            recent_chat_messages=["[10:00:01] user1: gogogo"],
            scene_description="Киберспортивный матч",
            detected_objects="person: 2",
            speech_history=["Раунд начался", "Заходим на точку!"],
            chat_stats={"positive_ratio": 0.9, "negative_ratio": 0.0},
            vlm_description="На табло 2:1",
        )

        self.assertIn("Красиво зашел", res.reply)
        sent_prompt = mock_post.call_args[1]["json"]["prompt"]
        self.assertIn("Киберспортивный матч", sent_prompt)
        self.assertIn("person: 2", sent_prompt)
        self.assertIn("визуальный анализ): На табло 2:1", sent_prompt)
        self.assertIn("Заходим на точку!", sent_prompt)
        self.assertIn("Раунд начался", sent_prompt)
        self.assertIn("user1: gogogo", sent_prompt)
        self.assertIn("Позитив: 90%", sent_prompt)

    def test_generate_reply_raises_when_ollama_unavailable(self):
        self.brain.is_ollama_available = False
        with self.assertRaises(RuntimeError):
            self.brain.generate_reply(
                streamer_speech="Тест", chat_mood="ПОЗИТИВ", visual_context=""
            )

    @patch("requests.post")
    def test_generate_reply_raises_on_http_error(self, mock_post):
        self.brain.is_ollama_available = True
        mock_post.return_value.status_code = 500
        mock_post.return_value.text = "Internal Server Error"
        with self.assertRaises(RuntimeError):
            self.brain.generate_reply(
                streamer_speech="Тест", chat_mood="ПОЗИТИВ", visual_context=""
            )


if __name__ == "__main__":
    unittest.main()
