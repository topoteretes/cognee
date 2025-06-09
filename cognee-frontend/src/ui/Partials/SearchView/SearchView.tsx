'use client';

import { v4 } from 'uuid';
import classNames from 'classnames';
import { useCallback, useEffect, useState } from 'react';
import { CTAButton, Stack, Text, DropdownSelect, TextArea, useBoolean, Input } from 'ohmy-ui';
import { fetch } from '@/utils';
import styles from './SearchView.module.css';
import getHistory from '@/modules/chat/getHistory';

interface Message {
  id: string;
  user: 'user' | 'system';
  text: any;
}

interface SelectOption {
  value: string;
  label: string;
}

export default function SearchView() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState<string>("");

  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(event.target.value);
  }, []);

  const searchOptions = [{
    value: 'GRAPH_COMPLETION',
    label: 'Completion using Cognee\'s graph based memory',
  }, {
    value: 'RAG_COMPLETION',
    label: 'Completion using RAG',
  }, {
    value: 'GRAPH_COMPLETION_COT',
    label: 'Cognee\'s Chain of Thought search',
  }, {
    value: 'GRAPH_COMPLETION_CONTEXT_EXTENSION',
    label: 'Cognee\'s Multi-Hop search',
  }];
  const [searchType, setSearchType] = useState(searchOptions[0]);
  const [rangeValue, setRangeValue] = useState(10);

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      const messagesContainerElement = document.getElementById('messages');
      if (messagesContainerElement) {
        const messagesElements = messagesContainerElement.children[0];

        if (messagesElements) {
          messagesContainerElement.scrollTo({
            top: messagesElements.scrollHeight,
            behavior: 'smooth',
          });
        }
      }
    }, 300);
  }, []);

  useEffect(() => {
    getHistory()
      .then((history) => {
        setMessages(history);
        scrollToBottom();
      });
  }, [scrollToBottom]);

  const handleSearchSubmit = useCallback((event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (inputValue.trim() === '') {
      return;
    }

    setMessages((currentMessages) => [
      ...currentMessages,
      {
        id: v4(),
        user: 'user',
        text: inputValue,
      },
    ]);

    scrollToBottom();

    setInputValue('');

    const searchTypeValue = searchType.value;

    fetch('/v1/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: inputValue.trim(),
        searchType: searchTypeValue,
        topK: rangeValue,
      }),
    })
      .then((response) => response.json())
      .then((systemMessage) => {
        setMessages((currentMessages) => [
          ...currentMessages,
          {
            id: v4(),
            user: 'system',
            text: convertToSearchTypeOutput(systemMessage, searchTypeValue),
          },
        ]);

        scrollToBottom();
      })
      .catch(() => {
        setInputValue(inputValue);
      });
  }, [inputValue, rangeValue, scrollToBottom, searchType.value]);

  const {
    value: isInputExpanded,
    setTrue: expandInput,
    setFalse: contractInput,
  } = useBoolean(false);

  const handleSubmitOnEnter = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      handleSearchSubmit(event as unknown as React.FormEvent<HTMLFormElement>);
    }
  };

  const handleRangeValueChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRangeValue(parseInt(event.target.value));
  };

  return (
    <Stack className={styles.searchViewContainer}>
      <DropdownSelect<SelectOption>
        value={searchType}
        options={searchOptions}
        onChange={setSearchType}
      />
      <div className={styles.messagesContainer} id="messages">
        <Stack gap="2" className={styles.messages} align="end">
          {messages.map((message) => (
            <Text
              key={message.id}
              className={classNames(styles.message, {
                [styles.userMessage]: message.user === "user",
              })}
            >
              {message?.text && (
                typeof(message.text) == "string" ? message.text : JSON.stringify(message.text)
              )}
            </Text>
          ))}
        </Stack>
      </div>
      <form onSubmit={handleSearchSubmit}>
        <Stack orientation="vertical" gap="2">
          <TextArea onKeyUp={handleSubmitOnEnter} style={{ transition: 'height 0.3s ease', height: isInputExpanded ? '128px' : '38px' }} onFocus={expandInput} onBlur={contractInput} value={inputValue} onChange={handleInputChange} name="searchInput" placeholder="Search" />
          <Stack orientation="horizontal" gap="between">
            <Stack orientation="horizontal" gap="2" align="center">
              <label><Text>Search range: </Text></label>
              <Input style={{ maxWidth: "90px" }} value={rangeValue} onChange={handleRangeValueChange} type="number" />
            </Stack>
            <CTAButton hugContent type="submit">Search</CTAButton>
          </Stack>
        </Stack>
      </form>
    </Stack>
  );
}


interface Node {
  name: string;
}

interface Relationship {
  relationship_name: string;
}

type InsightMessage = [Node, Relationship, Node];


function convertToSearchTypeOutput(systemMessages: any[], searchType: string): string {
  if (systemMessages.length > 0 && typeof(systemMessages[0]) === "string") {
    return systemMessages[0];
  }

  switch (searchType) {
    case 'INSIGHTS':
      return systemMessages.map((message: InsightMessage) => {
        const [node1, relationship, node2] = message;
        if (node1.name && node2.name) {
          return `${node1.name} ${relationship.relationship_name} ${node2.name}.`;
        }
        return '';
      }).join('\n');
    case 'SUMMARIES':
      return systemMessages.map((message: { text: string }) => message.text).join('\n');
    case 'CHUNKS':
      return systemMessages.map((message: { text: string }) => message.text).join('\n');
    default:
      return "";
  }
}
