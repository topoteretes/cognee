import { v4 } from "uuid";
import { useCallback, useState } from "react";
import { fetch, useBoolean } from "@/utils";
import { Dataset } from "@/modules/ingestion/useDatasets";

interface ChatMessage {
  id: string;
  user: "user" | "system";
  text: string;
}

const fetchMessages = () => {
  return fetch("/v1/search/")
    .then(response => response.json());
};

const sendMessage = (message: string, searchType: string) => {
  return fetch("/v1/search/", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: message,
      searchType,
      datasets: ["main_dataset"],
    }),
  })
    .then(response => response.json());
};

// Will be used in the future.
// eslint-disable-next-line @typescript-eslint/no-unused-vars
export default function useChat(dataset: Dataset) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);

  const {
    value: isSearchRunning,
    setTrue: disableSearchRun,
    setFalse: enableSearchRun,
  } = useBoolean(false);
  
  const refreshChat = useCallback(async () => {
    const data = await fetchMessages();
    return setMessages(data);
  }, []);

  const handleMessageSending = useCallback((message: string, searchType: string) => {
    const sentMessageId = v4();

    setMessages((messages) => [
      ...messages,
      {
        id: sentMessageId,
        user: "user",
        text: message,
      },
    ]);

    disableSearchRun();

    return sendMessage(message, searchType)
      .then(newMessages => {
        setMessages((messages) => [
          ...messages,
          ...newMessages.map((newMessage: string | []) => ({
            id: v4(),
            user: "system",
            text: convertToSearchTypeOutput(newMessage, searchType),
          })),
        ]);
      })
      .catch(() => {
        setMessages(
          (messages) => messages.filter(message => message.id !== sentMessageId),
        );
        throw new Error("Failed to send message. Please try again. If the issue persists, please contact support.")
      })
      .finally(() => enableSearchRun());
  }, [disableSearchRun, enableSearchRun]);

  return {
    messages,
    refreshChat,
    sendMessage: handleMessageSending,
    isSearchRunning,
  };
}


interface Node {
  name: string;
}

interface Relationship {
  relationship_name: string;
}

type InsightMessage = [Node, Relationship, Node];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function convertToSearchTypeOutput(systemMessage: any[] | any, searchType: string): string {
  if (Array.isArray(systemMessage) && systemMessage.length === 1 && typeof(systemMessage[0]) === "string") {
    return systemMessage[0];
  }

  switch (searchType) {
    case "INSIGHTS":
      return systemMessage.map((message: InsightMessage) => {
        const [node1, relationship, node2] = message;
        if (node1.name && node2.name) {
          return `${node1.name} ${relationship.relationship_name} ${node2.name}.`;
        }
        return "";
      }).join("\n");
    case "SUMMARIES":
      return systemMessage.map((message: { text: string }) => message.text).join("\n");
    case "CHUNKS":
      return systemMessage.map((message: { text: string }) => message.text).join("\n");
    default:
      return systemMessage;
  }
}
