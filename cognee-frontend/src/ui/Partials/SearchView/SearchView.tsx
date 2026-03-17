"use client";

import classNames from "classnames";
import { useCallback, useEffect, useRef, useState } from "react";

import { LoadingIndicator } from "@/ui/App";
import { CTAButton, Select, TextArea, Input } from "@/ui/elements";
import useChat from "@/modules/chat/hooks/useChat";

import styles from "./SearchView.module.css";

interface SelectOption {
  value: string;
  label: string;
}

interface SearchFormPayload extends HTMLFormElement {
  chatInput: HTMLInputElement;
}

const MAIN_DATASET = {
  id: "",
  data: [],
  status: "",
  name: "main_dataset",
};

export default function SearchView() {
  const searchOptions: SelectOption[] = [{
    value: "GRAPH_COMPLETION",
    label: "GraphRAG Completion",
  }, {
    value: "RAG_COMPLETION",
    label: "RAG Completion",
  }];

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      const messagesContainerElement = document.getElementById("messages");
      if (messagesContainerElement) {
        const messagesElements = messagesContainerElement.children[0];

        if (messagesElements) {
          messagesContainerElement.scrollTo({
            top: messagesElements.scrollHeight,
            behavior: "smooth",
          });
        }
      }
    }, 300);
  }, []);

  // Hardcoded to `main_dataset` for now, change when multiple datasets are supported.
  const { messages, refreshChat, sendMessage, isSearchRunning } = useChat(MAIN_DATASET);

  useEffect(() => {
    refreshChat()
      .then(() => scrollToBottom());
  }, [refreshChat, scrollToBottom]);

  const [searchInputValue, setSearchInputValue] = useState("");
  // Add state for top_k
  const [topK, setTopK] = useState(10);

  const handleSearchInputChange = useCallback((value: string) => {
    setSearchInputValue(value);
  }, []);

  // Add handler for top_k input
  const handleTopKChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    let value = parseInt(e.target.value, 10);
    if (isNaN(value)) value = 10;
    if (value < 1) value = 1;
    if (value > 100) value = 100;
    setTopK(value);
  }, []);

  const handleChatMessageSubmit = useCallback((event: React.FormEvent<SearchFormPayload>) => {
    event.preventDefault();

    const formElements = event.currentTarget;
    const searchType = formElements.searchType.value;

    const chatInput = searchInputValue.trim();

    if (chatInput === "") {
      return;
    }

    scrollToBottom();

    setSearchInputValue("");

    // Pass topK to sendMessage
    sendMessage(chatInput, searchType, topK)
      .then(scrollToBottom)
  }, [scrollToBottom, sendMessage, searchInputValue, topK]);

  const chatFormRef = useRef<HTMLFormElement>(null);

  const handleSubmitOnEnter = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      chatFormRef.current?.requestSubmit();
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-500 p-6 pt-16 rounded-3xl border-indigo-600 border-2 shadow-xl">
      <form onSubmit={handleChatMessageSubmit} ref={chatFormRef} className="flex flex-col gap-4 h-full">
        <div className="h-full overflow-y-auto" id="messages">
          <div className="flex flex-col gap-2 items-end justify-end min-h-full px-6 pb-4">
            {messages.map((message) => (
              <p
                key={message.id}
                className={classNames({
                  [classNames("ml-12 px-6 py-4 bg-gray-300 rounded-xl", styles.userMessage)]: message.user === "user",
                  [classNames("text-gray-200", styles.systemMessage)]: message.user !== "user",
                })}
              >
                {message?.text && (
                  typeof(message.text) == "string" ? message.text : JSON.stringify(message.text)
                )}
              </p>
            ))}
          </div>
        </div>

        <div className="p-4 bg-gray-300 rounded-xl flex flex-col gap-2">
          <TextArea
            value={searchInputValue}
            onChange={handleSearchInputChange}
            onKeyUp={handleSubmitOnEnter}
            isAutoExpanding
            name="chatInput"
            placeholder="Ask anything"
            contentEditable={true}
            className="resize-none min-h-14 max-h-96 overflow-y-auto"
          />
          <div className="flex flex-row items-center justify-between gap-4">
            <div className="flex flex-row items-center gap-4">
              <div className="flex flex-row items-center gap-2">
                <label className="text-gray-600 whitespace-nowrap">Search type:</label>
                <Select name="searchType" defaultValue={searchOptions[0].value} className="max-w-2xs">
                  {searchOptions.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </Select>
              </div>
              <div className="flex flex-row items-center gap-2">
                <label className="text-gray-600 whitespace-nowrap" title="Controls how many results to return. Smaller = focused, larger = broader graph exploration.">
                  Max results:
                </label>
                <Input
                  type="number"
                  name="topK"
                  min={1}
                  max={100}
                  value={topK}
                  onChange={handleTopKChange}
                  className="w-20"
                  title="Controls how many results to return. Smaller = focused, larger = broader graph exploration."
                />
              </div>
            </div>
            <CTAButton disabled={isSearchRunning} type="submit">
              {isSearchRunning? "Searching..." : "Search"}
              {isSearchRunning && <LoadingIndicator />}
            </CTAButton>
          </div>
        </div>
      </form>
    </div>
  );
}
