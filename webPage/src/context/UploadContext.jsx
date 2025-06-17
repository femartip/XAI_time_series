import { createContext, useContext, useReducer, useEffect, useRef } from 'react';


/*
Want to have global state for all the app. This allows to have the progress bar globally and be able to check its status.
The context is needed for that, now dont need to mount/unmount every time user navigates between sites.
*/

const UploadContext = createContext();

const uploadReducer = (state, action) => {
    switch (action.type) {
        case 'START_UPLOAD':
            return {
                ...state,
                [action.taskId]: {
                    taskId: action.taskId,
                    datasetName: action.datasetName,
                    status: 'uploading',
                    progress: 0,
                    message: 'Starting upload...',
                    isActive: true,
                    startTime: new Date()
                }
            };
        case 'SET_SERVER_TASK_ID':
            console.log('SET_SERVER_TASK_ID:', { clientTaskId: action.clientTaskId, serverTaskId: action.serverTaskId, currentState: Object.keys(state) });
            const { [action.clientTaskId]: clientUpload, ...restState } = state;
            return {
                ...restState,
                [action.serverTaskId]: {
                    ...clientUpload,
                    taskId: action.serverTaskId
                }
            };
        case 'UPDATE_PROGRESS':
            return {
                ...state,
                [action.taskId]: {
                    ...state[action.taskId],
                    ...action.data,
                    lastUpdated: new Date()
                }
            };
        case 'COMPLETE_UPLOAD':
            return {
                ...state,
                [action.taskId]: {
                    ...state[action.taskId],
                    isActive: false,
                    status: action.status || 'completed',
                    completedAt: new Date()
                }
            };
        case 'REMOVE_UPLOAD':
            const newState = { ...state };
            delete newState[action.taskId];
            return newState;
        default:
            return state;
    }
};

export const UploadProvider = ({ children }) => {
    const [uploads, dispatch] = useReducer(uploadReducer, {});
    const websocketRefs = useRef({});

    useEffect(() => {
        console.log('Uploads state:', uploads);
        const activeUploads = Object.values(uploads).filter(upload =>
            upload.status === 'uploading' &&
            upload.taskId &&
            !upload.taskId.startsWith('upload_') // Only connect with server taskId
        );

        console.log('Uploads to connect:', activeUploads.map(u => u.taskId));

        // Connect to new uploads that don't have websockets yet
        activeUploads.forEach(upload => {
            if (!websocketRefs.current[upload.taskId]) {
                console.log('🔗 Connecting WebSocket for:', upload.taskId);
                const ws = new WebSocket(`ws://localhost:8000/ws/progress/${upload.taskId}`);

                ws.onopen = () => {
                    console.log('✅ WebSocket connected for:', upload.taskId);
                };

                ws.onmessage = (event) => {
                    console.log('📨 Message received for', upload.taskId, ':', event.data);
                    try {
                        const data = JSON.parse(event.data);
                        console.log('📊 Parsed data:', data);

                        if (data.status === 'completed' || data.status === 'error') {
                            console.log('✅ Server says upload finished:', upload.taskId, data.status);
                            dispatch({ type: 'COMPLETE_UPLOAD', taskId: upload.taskId, status: data.status });
                        } else {
                            console.log('🔄 Updating progress:', upload.taskId, data);
                            dispatch({ type: 'UPDATE_PROGRESS', taskId: upload.taskId, data });
                        }
                    } catch (error) {
                        console.error('❌ Error parsing message:', error, event.data);
                    }
                };

                ws.onclose = (event) => {
                    console.log('🔌 WebSocket closed for:', upload.taskId, 'Code:', event.code);
                    delete websocketRefs.current[upload.taskId];
                };

                ws.onerror = (error) => {
                    console.error('💥 WebSocket error for:', upload.taskId, error);
                };

                websocketRefs.current[upload.taskId] = ws;
            }
        });

        // Clean up connections for uploads that are no longer active
        Object.keys(websocketRefs.current).forEach(taskId => {
            const upload = uploads[taskId];
            if (!upload || upload.status !== 'uploading') {
                console.log('🧹 Closing WebSocket for completed/removed upload:', taskId);
                const ws = websocketRefs.current[taskId];
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
                delete websocketRefs.current[taskId];
            }
        });

        // Cleanup function only runs when component unmounts
        return () => {
            console.log('🧹 Component unmounting - closing all WebSockets');
            Object.values(websocketRefs.current).forEach(ws => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.close();
                }
            });
            websocketRefs.current = {};
        };
    }, [uploads]);
};

export const useUpload = () => {
    const context = useContext(UploadContext);
    if (!context) {
        throw new Error('useUpload must be used within UploadProvider');
    }
    return context;
};
