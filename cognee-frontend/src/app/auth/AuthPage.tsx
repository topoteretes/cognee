import { TextLogo } from "@/ui/App";
import { Divider } from "@/ui/Layout";
import Footer from "@/ui/Partials/Footer/Footer";
import SignInForm from "@/ui/Partials/SignInForm/SignInForm";

export default function AuthPage() {
  return (
    <main className="flex flex-col h-full">
      <div className="pt-6 pr-3 pb-3 pl-6">
        <TextLogo width={86} height={24} />
      </div>
      <Divider />
      <div className="w-full max-w-md pt-12 pb-6 m-auto">
        <div className="flex flex-col w-full gap-8">
          <h1><span className="text-xl">Sign in</span></h1>
          <SignInForm />
        </div>
      </div>
      <div className="pl-6 pr-6">
        <Footer />
      </div>
    </main>
  )
}
