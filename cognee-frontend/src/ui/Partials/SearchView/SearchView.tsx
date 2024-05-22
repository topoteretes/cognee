import { CTAButton, CloseIcon, GhostButton, Input, Spacer, Stack, Text, DropdownSelect } from 'ohmy-ui';
import styles from './SearchView.module.css';
import { useCallback, useState } from 'react';
import { v4 } from 'uuid';
import classNames from 'classnames';

interface SearchViewProps {
  onClose: () => void;
}

interface Message {
  id: string;
  user: 'user' | 'system';
  text: string;
}

export default function SearchView({ onClose }: SearchViewProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState<string>("");

  const handleInputChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(event.target.value);
  }, []);

  const searchOptions = [{
    value: 'SIMILARITY',
    label: 'Similarity',
  }, {
    value: 'NEIGHBOR',
    label: 'Neighbor',
  }, {
    value: 'SUMMARY',
    label: 'Summary',
  }, {
    value: 'ADJACENT',
    label: 'Adjacent',
  }, {
    value: 'CATEGORIES',
    label: 'Categories',
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

  return (
    <Stack className={styles.searchViewContainer}>
      <Stack gap="between" align="center/" orientation="horizontal">
        <Spacer horizontal="2">
          <Text>Search</Text>
        </Spacer>
        <GhostButton onClick={onClose}>
          <CloseIcon />
        </GhostButton>
      </Stack>
      <Stack className={styles.searchContainer}>
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
          <Stack orientation="horizontal" gap="2">
            <DropdownSelect
              value={searchType}
              options={searchOptions}
              onChange={setSearchType}
            />
            <Input value={inputValue} onChange={handleInputChange} name="searchInput" placeholder="Search" />
            <CTAButton type="submit">Search</CTAButton>
          </Stack>
        </form>
      </Stack>
    </Stack>
  );
}
