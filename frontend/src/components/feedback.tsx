import {
  createContext,
  useContext,
  useState,
  type PropsWithChildren,
  type ReactNode,
} from "react";
import * as Toast from "@radix-ui/react-toast";
import * as Tooltip from "@radix-ui/react-tooltip";

type Notice = { title: string; detail?: string };

const ToastContext = createContext<(notice: Notice) => void>(() => undefined);

export function FeedbackProvider({ children }: PropsWithChildren) {
  const [notice, setNotice] = useState<Notice | null>(null);
  return (
    <Tooltip.Provider delayDuration={250}>
      <Toast.Provider swipeDirection="right">
        <ToastContext.Provider value={setNotice}>{children}</ToastContext.Provider>
        <Toast.Root
          className="toast-root"
          open={Boolean(notice)}
          onOpenChange={(open) => !open && setNotice(null)}
        >
          <Toast.Title className="font-medium text-slate-900">{notice?.title}</Toast.Title>
          {notice?.detail && <Toast.Description className="mt-1 text-sm text-slate-600">{notice.detail}</Toast.Description>}
        </Toast.Root>
        <Toast.Viewport className="toast-viewport" />
      </Toast.Provider>
    </Tooltip.Provider>
  );
}

export function useToast() {
  return useContext(ToastContext);
}

export function Hint({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Tooltip.Root>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="tooltip-content" sideOffset={8}>
          {label}
          <Tooltip.Arrow className="fill-slate-800" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}
