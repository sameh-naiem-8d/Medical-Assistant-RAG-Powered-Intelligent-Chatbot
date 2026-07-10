const { AiChatSession, AiChatMessage } = require("../models/ai-chat.model");

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || "http://127.0.0.1:8010";
const AI_SERVICE_TIMEOUT_MS = Number(process.env.AI_SERVICE_TIMEOUT_MS || 30000);

function userIdFromRequest(req) {
  return req.user?._id || req.user?.id;
}

async function callAiService(payload) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), AI_SERVICE_TIMEOUT_MS);
  try {
    const response = await fetch(`${AI_SERVICE_URL}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`AI service returned ${response.status}`);
    }
    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

exports.chatWithMedBridgeAI = async (req, res, next) => {
  try {
    const userId = userIdFromRequest(req);
    const { message, conversationId, source = "web", language } = req.body;

    if (!message || !message.trim()) {
      return res.status(400).json({ message: "message is required" });
    }

    let session = conversationId
      ? await AiChatSession.findOne({ _id: conversationId, userId, status: "active" })
      : null;

    if (!session) {
      session = await AiChatSession.create({
        userId,
        source,
        language: language || "ar",
        title: message.slice(0, 60),
      });
    }

    const recentMessages = await AiChatMessage.find({
      sessionId: session._id,
      userId,
      isVisibleToUser: true,
      role: { $in: ["user", "assistant"] },
    })
      .sort({ createdAt: -1 })
      .limit(10)
      .lean();

    const history = recentMessages
      .reverse()
      .map((item) => ({ role: item.role, content: item.content }));

    await AiChatMessage.create({
      sessionId: session._id,
      userId,
      role: "user",
      content: message,
      language: language || session.language,
      isVisibleToUser: true,
    });

    const aiResponse = await callAiService({
      user_id: String(userId),
      conversation_id: String(session._id),
      language: language || session.language,
      source,
      message,
      history,
    });

    await AiChatMessage.create({
      sessionId: session._id,
      userId,
      role: "assistant",
      content: aiResponse.answer,
      language: aiResponse.case_state_update?.language || language || session.language,
      isVisibleToUser: true,
      aiMetadata: {
        mode: aiResponse.mode,
        urgency: aiResponse.urgency_level,
        possibleDiagnosis: aiResponse.possible_diagnosis,
        displayDiagnosisAr: aiResponse.display_diagnosis_ar,
        doctorSuggestion: aiResponse.suggested_doctor,
        displayDoctorAr: aiResponse.display_doctor_ar,
        followUpQuestions: aiResponse.follow_up_questions,
        safetyFlags: aiResponse.case_state_update?.safety_flags || [],
        confidence: aiResponse.confidence,
      },
    });

    session.lastKnownCaseState = aiResponse.case_state_update || {};
    session.lastMessageAt = new Date();
    await session.save();

    return res.json({
      conversationId: String(session._id),
      answer: aiResponse.answer,
      mode: aiResponse.mode,
      urgencyLevel: aiResponse.urgency_level,
      followUpQuestions: aiResponse.follow_up_questions,
      displayDiagnosisAr: aiResponse.display_diagnosis_ar,
      displayDoctorAr: aiResponse.display_doctor_ar,
    });
  } catch (error) {
    return next(error);
  }
};

exports.listAiChats = async (req, res, next) => {
  try {
    const sessions = await AiChatSession.find({ userId: userIdFromRequest(req) }).sort({ lastMessageAt: -1 });
    return res.json({ sessions });
  } catch (error) {
    return next(error);
  }
};

exports.getAiChat = async (req, res, next) => {
  try {
    const userId = userIdFromRequest(req);
    const session = await AiChatSession.findOne({ _id: req.params.id, userId });
    if (!session) return res.status(404).json({ message: "Chat not found" });
    const messages = await AiChatMessage.find({ sessionId: session._id, userId, isVisibleToUser: true }).sort({ createdAt: 1 });
    return res.json({ session, messages });
  } catch (error) {
    return next(error);
  }
};

exports.archiveAiChat = async (req, res, next) => {
  try {
    const session = await AiChatSession.findOneAndUpdate(
      { _id: req.params.id, userId: userIdFromRequest(req) },
      { status: "archived" },
      { new: true }
    );
    if (!session) return res.status(404).json({ message: "Chat not found" });
    return res.json({ session });
  } catch (error) {
    return next(error);
  }
};
