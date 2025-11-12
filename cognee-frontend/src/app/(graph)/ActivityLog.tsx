"use client";

import { forwardRef, useCallback, useImperativeHandle, useState } from "react";

type ActivityLog = {
  id: string;
  timestamp: number;
  activity: string;
};

export interface ActivityLogAPI {
  updateActivityLog: (activityLog: ActivityLog[]) => void;
}

const formatter = new Intl.DateTimeFormat("en-GB", { dateStyle: "short", timeStyle: "medium" });

const ActivityLog = forwardRef<ActivityLogAPI>(function ActivityLog(_, ref) {
  const [activityLog, updateActivityLog] = useState<ActivityLog[]>([]);

  const handleActivityLogUpdate = useCallback(
    (newActivities: ActivityLog[]) => {
      updateActivityLog([...activityLog, ...newActivities]);

      const activityLogContainer = document.getElementById("activityLogContainer");

      if (activityLogContainer) {
        activityLogContainer.scrollTo({ top: 0, behavior: "smooth" });
      }
    },
    [activityLog],
  );

  useImperativeHandle(ref, () => ({
    updateActivityLog: handleActivityLogUpdate,
  }));

  return (
    <div className="overflow-y-auto max-h-96" id="activityLogContainer">
      <div className="flex flex-col-reverse gap-2">
        {activityLog.map((activity) => (
          <div key={activity.id} className="flex gap-2 items-top">
            <span className="flex-1/3 text-xs text-gray-300 whitespace-nowrap mt-1.5">{formatter.format(activity.timestamp)}: </span>
            <span className="flex-2/3 text-white  whitespace-normal">{activity.activity}</span>
          </div>
        ))}
        {!activityLog.length && <span className="text-white">No activity logged.</span>}
      </div>
    </div>
  );
});

export default ActivityLog;
