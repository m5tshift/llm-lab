"""Unit tests for Twitch IRC message parsing and color assignments."""

import unittest
from src.connectors.twitch_chat import TwitchChatListener, get_twitch_color


class TestTwitchChatParsing(unittest.TestCase):
    def test_get_twitch_color_deterministic(self):
        col1 = get_twitch_color("gamer123")
        col2 = get_twitch_color("gamer123")
        self.assertEqual(col1, col2)
        self.assertTrue(col1.startswith("#"))

    def test_get_twitch_color_custom_preserved(self):
        self.assertEqual(get_twitch_color("gamer", "#123456"), "#123456")

    def test_parse_privmsg_tagged(self):
        listener = TwitchChatListener("testchannel")
        raw_line = "@badge-info=;badges=turbo/1;color=#00FF7F;display-name=CoolViewer;emotes=;id=123 :coolviewer!coolviewer@tmi.twitch.tv PRIVMSG #testchannel :POGCHAMP WHAT A PLAY"
        listener._parse_privmsg(raw_line)

        messages = listener.get_recent_chat_objects()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].sender, "CoolViewer")
        self.assertEqual(messages[0].text, "POGCHAMP WHAT A PLAY")
        self.assertEqual(messages[0].color, "#00FF7F")

    def test_parse_privmsg_classic(self):
        listener = TwitchChatListener("testchannel")
        raw_line = ":classic_user!classic_user@classic_user.tmi.twitch.tv PRIVMSG #testchannel :hello stream"
        listener._parse_privmsg(raw_line)

        messages = listener.get_recent_chat_objects()
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].sender, "classic_user")
        self.assertEqual(messages[0].text, "hello stream")


if __name__ == "__main__":
    unittest.main()
