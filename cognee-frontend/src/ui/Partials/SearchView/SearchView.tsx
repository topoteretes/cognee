import { v4 } from 'uuid';
import classNames from 'classnames';
import { useCallback, useState } from 'react';
import { CTAButton, Stack, Text, DropdownSelect, TextArea, useBoolean } from 'ohmy-ui';
import styles from './SearchView.module.css';

interface Message {
  id: string;
  user: 'user' | 'system';
  text: string;
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
    value: 'SIMILARITY',
    label: 'Look for similar graph nodes',
  }, {
    value: 'SUMMARY',
    label: 'Get a summary related to query',
  }, {
    value: 'ADJACENT',
    label: 'Look for graph node\'s neighbors',
  }, {
    value: 'TRAVERSE',
    label: 'Traverse through the graph and get knowledge',
  }];
  const [searchType, setSearchType] = useState(searchOptions[0]);

  const handleSearchSubmit = useCallback((event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    setMessages((currentMessages) => [
      ...currentMessages,
      {
        id: v4(),
        user: 'user',
        text: inputValue,
      },
    ]);

    fetch('http://localhost:8000/search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query_params: {
          query: inputValue,
          searchType: searchType.value,
        },
      }),
    })
      .then((response) => response.json())
      .then((systemMessage) => {
        setMessages((currentMessages) => [
          ...currentMessages,
          {
            id: v4(),
            user: 'system',
            text: systemMessage,
          },
        ]);
        setInputValue('');
      })
  }, [inputValue, searchType]);

  const {
    value: isInputExpanded,
    setTrue: expandInput,
    setFalse: contractInput,
  } = useBoolean(false);

  return (
    <Stack className={styles.searchViewContainer}>
      <DropdownSelect<SelectOption>
        value={searchType}
        options={searchOptions}
        onChange={setSearchType}
      />
      <div className={styles.messagesContainer}>
        <Stack gap="2" className={styles.messages} align="end">
          {messages.map((message) => (
            <Text
              key={message.id}
              className={classNames(styles.message, {
                [styles.userMessage]: message.user === "user",
              })}
            >
              {message.text}
            </Text>
          ))}
        </Stack>
      </div>
      <form onSubmit={handleSearchSubmit}>
        <Stack orientation="horizontal" align="end/" gap="2">
          <TextArea style={{ height: isInputExpanded ? '128px' : '38px' }} onFocus={expandInput} onBlur={contractInput} value={inputValue} onChange={handleInputChange} name="searchInput" placeholder="Search" />
          <CTAButton hugContent type="submit">Search</CTAButton>
        </Stack>
      </form>
    </Stack>
  );
}
