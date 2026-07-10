const mongoose = require("mongoose");

const aiChatSessionSchema = new mongoose.Schema(
  {
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    title: { type: String, default: "MedBridge chat" },
    language: { type: String, default: "ar" },
    source: { type: String, enum: ["mobile", "web", "local_demo"], default: "web" },
    status: { type: String, enum: ["active", "archived"], default: "active" },
    summary: { type: String },
    lastKnownCaseState: { type: mongoose.Schema.Types.Mixed, default: {} },
    lastMessageAt: { type: Date, default: Date.now },
  },
  { timestamps: true }
);

const aiChatMessageSchema = new mongoose.Schema(
  {
    sessionId: { type: mongoose.Schema.Types.ObjectId, ref: "AiChatSession", required: true, index: true },
    userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true, index: true },
    role: { type: String, enum: ["user", "assistant", "system"], required: true },
    content: { type: String, required: true },
    language: { type: String, default: "ar" },
    isVisibleToUser: { type: Boolean, default: true },
    aiMetadata: { type: mongoose.Schema.Types.Mixed, default: {} },
  },
  { timestamps: true }
);

module.exports = {
  AiChatSession: mongoose.model("AiChatSession", aiChatSessionSchema),
  AiChatMessage: mongoose.model("AiChatMessage", aiChatMessageSchema),
};
