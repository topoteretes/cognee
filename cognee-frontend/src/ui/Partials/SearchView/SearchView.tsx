import { CTAButton, CloseIcon, GhostButton, Input, Spacer, Stack, Text } from 'ohmy-ui';
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
  }, [inputValue]);
  
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
        <form onSubmit={handleSearchSubmit}>
          <Stack orientation="horizontal" gap="2">
            <Input value={inputValue} onChange={handleInputChange} name="searchInput" placeholder="Search" />
            <CTAButton type="submit">Search</CTAButton>
          </Stack>
        </form>
      </Stack>
    </Stack>
  );
}
