import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const investigate = async (file, accountId, hopRadius = 2, windowDays = 30) => {
  const form = new FormData()
  form.append('file', file)
  const { data } = await api.post(
    `/investigate/v3?account_id=${accountId}&hop_radius=${hopRadius}&time_window_days=${windowDays}`,
    form, { headers: { 'Content-Type': 'multipart/form-data' } }
  )
  return data
}

export default api