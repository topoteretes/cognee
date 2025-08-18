// import { useCallback, useEffect, useState } from 'react';
// import {
//   CTAButton,
//   DropdownSelect,
//   FormGroup,
//   FormInput,
//   FormLabel,
//   Input,
//   Spacer,
//   Stack,
//   useBoolean,
// } from 'ohmy-ui';
// import { LoadingIndicator } from '@/ui/App';
// import { fetch } from '@/utils';

// interface SelectOption {
//   label: string;
//   value: string;
// }

// interface SettingsForm extends HTMLFormElement {
//   vectorDBUrl: HTMLInputElement;
//   vectorDBApiKey: HTMLInputElement;
//   llmProvider: HTMLInputElement;
//   llmModel: HTMLInputElement;
//   llmApiKey: HTMLInputElement;
//   llmEndpoint: HTMLInputElement;
//   llmApiVersion: HTMLInputElement;
// }

// const defaultProvider = {
//   label: 'OpenAI',
//   value: 'openai',
// };

// const defaultModel = {
//   label: 'gpt-5-mini',
//   value: 'gpt-5-mini',
// };

// export default function Settings({ onDone = () => {}, submitButtonText = 'Save' }) {
//   const [llmConfig, setLLMConfig] = useState<{
//     apiKey: string;
//     model: string;
//     endpoint: string;
//     apiVersion: string;
//     provider: string;
//   }>();
//   const [vectorDBConfig, setVectorDBConfig] = useState<{
//     url: string;
//     apiKey: string;
//     provider: SelectOption;
//     providers: SelectOption[];
//   }>();

//   const {
//     value: isSaving,
//     setTrue: startSaving,
//     setFalse: stopSaving,
//   } = useBoolean(false);

//   const saveConfig = (event: React.FormEvent<SettingsForm>) => {
//     event.preventDefault();
//     const formElements = event.currentTarget;

//     const newVectorConfig = {
//       provider: vectorDBConfig?.provider.value,
//       url: formElements.vectorDBUrl.value,
//       apiKey: formElements.vectorDBApiKey.value,
//     };

//     const newLLMConfig = {
//       provider: formElements.llmProvider.value,
//       model: formElements.llmModel.value,
//       apiKey: formElements.llmApiKey.value,
//       endpoint: formElements.llmEndpoint.value,
//       apiVersion: formElements.llmApiVersion.value,
//     };

//     startSaving();

//     fetch('/v1/settings', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({
//         llm: newLLMConfig,
//         vectorDb: newVectorConfig,
//       }),
//     })
//       .then(() => {
//         onDone();
//       })
//       .finally(() => stopSaving());
//   };

//   const handleVectorDBChange = useCallback((newVectorDBProvider: SelectOption) => {
//     setVectorDBConfig((config) => {
//       if (config?.provider !== newVectorDBProvider) {
//         return {
//          ...config,
//           providers: config?.providers || [],
//           provider: newVectorDBProvider,
//           url: '',
//           apiKey: '',
//         };
//       }
//       return config;
//     });
//   }, []);

//   useEffect(() => {
//     const fetchConfig = async () => {
//       const response = await fetch('/v1/settings');
//       const settings = await response.json();

//       if (!settings.llm.provider) {
//         settings.llm.provider = settings.llm.providers[0].value;
//       }
//       if (!settings.llm.model) {
//         settings.llm.model = settings.llm.models[settings.llm.provider][0].value;
//       }
//       if (!settings.vectorDb.provider) {
//         settings.vectorDb.provider = settings.vectorDb.providers[0];
//       } else {
//         settings.vectorDb.provider = settings.vectorDb.providers.find((provider: SelectOption) => provider.value === settings.vectorDb.provider);
//       }
//       setLLMConfig(settings.llm);
//       setVectorDBConfig(settings.vectorDb);
//     };
//     fetchConfig();
//   }, []);

//   return (
//     <form onSubmit={saveConfig} style={{ width: "100%", overflowY: "auto", maxHeight: "500px" }}>
//       <Stack gap="4" orientation="vertical">
//         <Stack gap="4" orientation="vertical">
//           <FormGroup orientation="vertical" align="center/" gap="2">
//             <FormLabel>LLM provider:</FormLabel>
//             <FormInput>
//               <Input defaultValue={llmConfig?.provider} name="llmProvider" placeholder="LLM provider" />
//             </FormInput>
//           </FormGroup>
//           <FormGroup orientation="vertical" align="center/" gap="2">
//             <FormLabel>LLM model:</FormLabel>
//             <FormInput>
//               <Input defaultValue={llmConfig?.model} name="llmModel" placeholder="LLM model" />
//             </FormInput>
//           </FormGroup>
//           <FormGroup orientation="vertical" align="center/" gap="2">
//             <FormLabel>LLM endpoint:</FormLabel>
//             <FormInput>
//               <Input defaultValue={llmConfig?.endpoint} name="llmEndpoint" placeholder="LLM endpoint url" />
//             </FormInput>
//           </FormGroup>
//           <FormGroup orientation="vertical" align="center/" gap="2">
//             <FormLabel>LLM API key:</FormLabel>
//             <FormInput>
//               <Input defaultValue={llmConfig?.apiKey} name="llmApiKey" placeholder="LLM API key" />
//             </FormInput>
//           </FormGroup>
//           <FormGroup orientation="vertical" align="center/" gap="2">
//             <FormLabel>LLM API version:</FormLabel>
//             <FormInput>
//               <Input defaultValue={llmConfig?.apiVersion} name="llmApiVersion" placeholder="LLM API version" />
//             </FormInput>
//           </FormGroup>
//         </Stack>

//         <Stack gap="2" orientation="vertical">
//           <FormGroup orientation="vertical" align="center/" gap="2">
//             <FormLabel>Vector DB provider:</FormLabel>
//             <DropdownSelect
//               value={vectorDBConfig?.provider || null}
//               options={vectorDBConfig?.providers || []}
//               onChange={handleVectorDBChange}
//             />
//           </FormGroup>
//           <FormInput>
//             <Input defaultValue={vectorDBConfig?.url} name="vectorDBUrl" placeholder="Vector DB instance url" />
//           </FormInput>
//           <FormInput>
//             <Input defaultValue={vectorDBConfig?.apiKey} name="vectorDBApiKey" placeholder="Vector DB API key" />
//           </FormInput>
//           <Stack align="/end">
//             <Spacer top="2">
//               <CTAButton type="submit">
//                 <Stack gap="2" orientation="vertical" align="center/">
//                   {submitButtonText}
//                   {isSaving && <LoadingIndicator />}
//                 </Stack>
//               </CTAButton>
//             </Spacer>
//           </Stack>
//         </Stack>
//       </Stack>
//     </form>
//   )
// }
