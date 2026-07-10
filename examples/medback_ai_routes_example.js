const express = require("express");
const router = express.Router();
const { protect, isUser } = require("../middleware/auth.middleware");
const { chatWithMedBridgeAI, listAiChats, getAiChat, archiveAiChat } = require("../controllers/ai-chat.controller");

router.use(protect);

router.post("/chat", isUser, chatWithMedBridgeAI);
router.get("/chats", isUser, listAiChats);
router.get("/chats/:id", isUser, getAiChat);
router.patch("/chats/:id/archive", isUser, archiveAiChat);

module.exports = router;
