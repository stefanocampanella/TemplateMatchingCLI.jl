import Base.+, Base.show, Base.isless


+(x::Dates.UTInstant{Microsecond}, d::Microsecond) = Dates.UTInstant(x.periods + d)

function show(io::IO, x::Dates.UTInstant{Microsecond})
    fpart, ipart = modf(1e-3 * x.periods.value)
    usec = round(Int, 1e3 * fpart)
    msec = convert(Int, ipart)
    date = convert(DateTime, Millisecond(msec))
    print(io, "$date$usec")
end

show(io::IO, ::MIME"text/plain", x::Dates.UTInstant{Microsecond}) = show(io, x)

function DateTimeMicrosecond(y, m, d, h, mi, s, us)
    dt = DateTime(y, m, d, h, mi, s)
    rata = Microsecond(us) + convert(Microsecond, dt.instant.periods)
    Dates.UTInstant(rata)
end

DateTimeMicrosecond(dt::DateTime) = Dates.UTInstant(convert(Microsecond, dt.instant.periods))

isless(t1::Dates.UTInstant{Microsecond}, t2::Dates.UTInstant{Microsecond}) = isless(t1.periods.value, t2.periods.value)
