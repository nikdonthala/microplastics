// NOTE: The real build steps live in vercel.json -> buildCommand.
// This file exists only to document the flow:
//   1. cd frontend && npm install && npm run build  -> frontend/dist
//   2. Vercel packages api/index.py as a Python function using api/requirements.txt
// No-op by design.
export default {};
